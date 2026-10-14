"""Bounded sequential O_DIRECT output for local checkpoint payloads."""

import ctypes
import errno
import fcntl
import logging
import mmap
import os
from pathlib import Path
from uuid import uuid4

log = logging.getLogger(__name__)
_BUFFER_BYTES = 1024 * 1024


class DirectSink:
    """An exclusively created temporary file with an aligned, bounded write buffer.

    ``tell()`` reports logical bytes, never alignment padding. ``flush()`` drains
    the buffer and truncates padding, but does not provide durability. It is safe
    to append after a flush, including the flushes issued by ``torch.save``.
    As with a delete=False NamedTemporaryFile, the caller owns the final
    flush/fdatasync/close/atomic rename sequence and must unlink the temporary
    on failure (or call ``abort()``). Closing an already flushed file performs
    no write or truncate. This object is sequential-only and not thread-safe.
    Only new files are opened, and there is no buffered fallback.
    Unsupported O_DIRECT requests fail explicitly. Memory use is one buffer
    plus O(page size), independent of the amount serialized.
    """

    def __init__(self, directory: str | os.PathLike):
        buffer_bytes = _BUFFER_BYTES
        self.alignment = os.sysconf("SC_PAGE_SIZE")
        if buffer_bytes <= 0 or buffer_bytes % self.alignment:
            raise ValueError("buffer_bytes must be a positive multiple of the page size")
        if not hasattr(os, "O_DIRECT") or not hasattr(os, "fdatasync"):
            raise OSError(errno.ENOTSUP, "O_DIRECT and fdatasync are required")
        self.buffer_bytes = buffer_bytes
        self.name = str(Path(directory) / f"direct-{uuid4().hex}.distcp")
        self.mode = "wb"
        self._buffer: mmap.mmap | None = None
        self._fd: int | None = None
        self._failed = False
        self._used = self._base = self._position = 0
        self._dirty = self._needs_truncate = False

        # Keep exclusive creation outside cleanup: a collision belongs to someone else.
        self._fd = os.open(
            self.name, os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_DIRECT | os.O_CLOEXEC, 0o600
        )
        try:
            if not fcntl.fcntl(self._fd, fcntl.F_GETFL) & os.O_DIRECT:
                raise OSError(errno.ENOTSUP, "filesystem did not retain O_DIRECT")
            self._buffer = mmap.mmap(-1, buffer_bytes)
            address = ctypes.addressof(ctypes.c_char.from_buffer(self._buffer))
            if address % self.alignment:
                raise OSError(errno.EINVAL, "unaligned direct-I/O buffer")
        except BaseException:
            self._abort_preserving_error()
            raise

    @property
    def closed(self) -> bool:
        return self._fd is None

    def _check_open(self) -> None:
        if self.closed:
            raise ValueError("I/O operation on closed direct file")
        if self._failed:
            raise OSError(errno.EIO, "direct file has failed; abort it instead of retrying")

    def fileno(self) -> int:
        self._check_open()
        assert self._fd is not None
        return self._fd

    def tell(self) -> int:
        self._check_open()
        return self._position

    def writable(self) -> bool:
        return not self.closed and not self._failed

    def seekable(self) -> bool:
        return False

    def readable(self) -> bool:
        return False

    def _write_pending(self) -> None:
        if not self._dirty:
            return
        assert self._buffer is not None and self._fd is not None
        padded = (self._used + self.alignment - 1) // self.alignment * self.alignment
        with memoryview(self._buffer) as view:
            view[self._used : padded] = b"\0" * (padded - self._used)
            done = 0
            while done < padded:
                with view[done:padded] as part:
                    try:
                        written = os.pwrite(self._fd, part, self._base + done)
                    except InterruptedError:
                        continue
                if (
                    not isinstance(written, int)
                    or written <= 0
                    or written > padded - done
                    or written % self.alignment
                ):
                    raise OSError(errno.EIO, f"invalid aligned write progress: {written}")
                done += written
        self._dirty = False
        self._needs_truncate = True

    def write(self, data) -> int:
        self._check_open()
        assert self._buffer is not None
        try:
            with memoryview(data).cast("B") as view:
                done = 0
                while done < len(view):
                    count = min(self.buffer_bytes - self._used, len(view) - done)
                    self._buffer[self._used : self._used + count] = view[done : done + count]
                    self._used += count
                    self._position += count
                    done += count
                    self._dirty = True
                    if self._used == self.buffer_bytes:
                        self._write_pending()
                        self._base += self._used
                        self._used = 0
                return done
        except BaseException:
            self._failed = True
            raise

    def flush(self) -> None:
        self._check_open()
        assert self._fd is not None and self._buffer is not None
        try:
            self._write_pending()
            if self._needs_truncate:
                os.ftruncate(self._fd, self._position)
                self._needs_truncate = False
            # Only the last partial page must be retained for aligned overwrite.
            complete = self._used // self.alignment * self.alignment
            tail = self._used - complete
            if complete:
                self._buffer[:tail] = self._buffer[complete : self._used]
                self._base += complete
                self._used = tail
        except BaseException:
            self._failed = True
            raise

    def _release(self) -> None:
        fd, self._fd = self._fd, None
        buffer, self._buffer = self._buffer, None
        try:
            if fd is not None:
                # Never retry close: the descriptor may already have been released.
                os.close(fd)
        finally:
            if buffer is not None:
                buffer.close()

    def close(self) -> None:
        """Flush and close, without sync or publication; failed files are not flushed."""
        if self.closed:
            return
        try:
            if not self._failed:
                self.flush()
        except BaseException:
            self._failed = True
            try:
                self._release()
            except OSError:
                log.warning("Error closing failed direct file %s", self.name, exc_info=True)
            raise
        try:
            self._release()
        except BaseException:
            self._failed = True
            raise

    def abort(self) -> None:
        """Close without writing and remove only this instance's temporary path."""
        self._failed = True
        try:
            self._release()
        finally:
            Path(self.name).unlink(missing_ok=True)

    def _abort_preserving_error(self) -> None:
        try:
            self.abort()
        except OSError:
            log.warning("Error cleaning direct temporary %s", self.name, exc_info=True)

    def __enter__(self) -> "DirectSink":  # noqa: PYI034 - also supports Python 3.10
        self._check_open()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if exc_type is not None:
            self._abort_preserving_error()
        else:
            try:
                self.close()
            except BaseException:
                self._abort_preserving_error()
                raise

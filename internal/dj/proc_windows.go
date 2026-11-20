//go:build windows

package dj

import (
	"os/exec"
	"syscall"
)

// hideWindow stops each ffmpeg child from flashing a console window when the
// bot runs without one (e.g. launched from a shortcut).
func hideWindow(cmd *exec.Cmd) {
	cmd.SysProcAttr = &syscall.SysProcAttr{HideWindow: true, CreationFlags: 0x08000000} // CREATE_NO_WINDOW
}

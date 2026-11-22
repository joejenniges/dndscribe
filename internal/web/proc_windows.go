//go:build windows

package web

import (
	"os/exec"
	"syscall"
)

// hideWindow stops a child process (the tailscale CLI) flashing a console
// window when the bot runs without one.
func hideWindow(cmd *exec.Cmd) {
	cmd.SysProcAttr = &syscall.SysProcAttr{HideWindow: true, CreationFlags: 0x08000000} // CREATE_NO_WINDOW
}

//go:build !windows

package dj

import "os/exec"

func hideWindow(cmd *exec.Cmd) {}

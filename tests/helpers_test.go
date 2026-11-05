package tests

import (
	"encoding/csv"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
)

func repoRoot(t *testing.T) string {
	t.Helper()

	wd, err := os.Getwd()
	if err != nil {
		t.Fatalf("get working directory: %v", err)
	}
	return filepath.Dir(wd)
}

func runGoCommand(t *testing.T, args ...string) []byte {
	t.Helper()

	cacheDir := filepath.Join(t.TempDir(), "gocache")
	cmd := exec.Command("go", args...)
	cmd.Dir = repoRoot(t)
	cmd.Env = append(os.Environ(), "GO111MODULE=on", "GOCACHE="+cacheDir)
	output, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("run %v: %v\n%s", args, err, string(output))
	}
	return output
}

func writeCSVFile(t *testing.T, path string, rows [][]string) {
	t.Helper()

	file, err := os.Create(path)
	if err != nil {
		t.Fatalf("create csv %s: %v", path, err)
	}
	defer file.Close()

	writer := csv.NewWriter(file)
	if err := writer.WriteAll(rows); err != nil {
		t.Fatalf("write csv %s: %v", path, err)
	}
	writer.Flush()
	if err := writer.Error(); err != nil {
		t.Fatalf("flush csv %s: %v", path, err)
	}
}

func readCSVFile(t *testing.T, path string) [][]string {
	t.Helper()

	file, err := os.Open(path)
	if err != nil {
		t.Fatalf("open csv %s: %v", path, err)
	}
	defer file.Close()

	rows, err := csv.NewReader(file).ReadAll()
	if err != nil {
		t.Fatalf("read csv %s: %v", path, err)
	}
	return rows
}

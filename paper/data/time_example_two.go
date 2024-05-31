package main

import (
	"strings"
	"time"
)

func weightTimeComplex(dt time.Time) int {
	x := 1
	y := 3
	z := 5

	if IsWeekend(dt.Weekday()) {
		x = 2
	}

	if dt.Hour() >= 12 {
		y = 4
	}

	weekday := strings.ToLower(dt.Weekday().String())
	if strings.Contains(weekday, "r") {
		z = 6
	}

	return x * y * z
}

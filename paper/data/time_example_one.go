package main

import (
	"time"
)

func weightTime(dt time.Time) int {
	x := 1
	y := 3

	if IsWeekend(dt.Weekday()) {
		x = 2
	}

	if dt.Hour() >= 12 {
		y = 4
	}

	return x * y
}

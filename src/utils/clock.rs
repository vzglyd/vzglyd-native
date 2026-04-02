//! Clock and time utilities.

use std::time::{SystemTime, UNIX_EPOCH};

/// Returns the current time in seconds since UNIX epoch.
pub fn now_secs() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

const MELBOURNE_STD_OFFSET_SECS: i32 = 10 * 60 * 60;
const MELBOURNE_DST_OFFSET_SECS: i32 = 11 * 60 * 60;

/// Returns the current Melbourne local time expressed as seconds since midnight.
pub fn melbourne_clock_seconds() -> f32 {
    let (_, _, _, hour, minute, second) = epoch_to_melbourne_components(now_unix_secs());
    f32::from(hour) * 3_600.0 + f32::from(minute) * 60.0 + f32::from(second)
}

fn now_unix_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0)
}

fn epoch_to_melbourne_components(epoch_secs: u64) -> (i32, u8, u8, u8, u8, u8) {
    let mut offset = MELBOURNE_STD_OFFSET_SECS;
    for _ in 0..2 {
        let shifted = (epoch_secs as i64 + i64::from(offset)) as u64;
        let (year, month, day, hour, minute, second) = utc_ymdhms_from_unix(shifted);
        let next_offset = melbourne_offset_seconds(year, month, day, hour);
        if next_offset == offset {
            return (year, month, day, hour, minute, second);
        }
        offset = next_offset;
    }
    utc_ymdhms_from_unix((epoch_secs as i64 + i64::from(offset)) as u64)
}

fn melbourne_offset_seconds(year: i32, month: u8, day: u8, hour: u8) -> i32 {
    let first_sunday_october = first_sunday(year, 10);
    let first_sunday_april = first_sunday(year, 4);
    let is_dst = if !(4..10).contains(&month) {
        true
    } else if (5..=9).contains(&month) {
        false
    } else if month == 10 {
        day > first_sunday_october || (day == first_sunday_october && hour >= 2)
    } else {
        day < first_sunday_april || (day == first_sunday_april && hour < 3)
    };

    if is_dst {
        MELBOURNE_DST_OFFSET_SECS
    } else {
        MELBOURNE_STD_OFFSET_SECS
    }
}

fn first_sunday(year: i32, month: u8) -> u8 {
    (1..=7)
        .find(|day| weekday_abbrev(year, month, *day) == "Sun")
        .unwrap_or(1)
}

fn utc_ymdhms_from_unix(epoch_secs: u64) -> (i32, u8, u8, u8, u8, u8) {
    let days = (epoch_secs / 86_400) as i64;
    let (year, month, day) = civil_from_days(days);
    let seconds_today = epoch_secs % 86_400;
    let hour = (seconds_today / 3_600) as u8;
    let minute = ((seconds_today / 60) % 60) as u8;
    let second = (seconds_today % 60) as u8;
    (year, month, day, hour, minute, second)
}

fn weekday_abbrev(year: i32, month: u8, day: u8) -> &'static str {
    const WEEKDAYS: [&str; 7] = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];
    let offsets = [0, 3, 2, 5, 0, 3, 5, 1, 4, 6, 2, 4];
    let mut year = year;
    if month < 3 {
        year -= 1;
    }
    let weekday = (year + year / 4 - year / 100
        + year / 400
        + offsets[month.saturating_sub(1) as usize]
        + i32::from(day))
        % 7;
    WEEKDAYS[weekday as usize]
}

fn civil_from_days(days: i64) -> (i32, u8, u8) {
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let year = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = mp + if mp < 10 { 3 } else { -9 };
    let year = year + if month <= 2 { 1 } else { 0 };
    (year as i32, month as u8, day as u8)
}

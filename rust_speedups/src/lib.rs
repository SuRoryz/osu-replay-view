use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyReadwriteArray2};
use pyo3::prelude::*;

fn locate_right(times: &[f64], time_ms: f64, anchor: usize) -> usize {
    if times.is_empty() {
        return 0;
    }
    let n = times.len();
    let mut lo;
    let mut hi;
    let clamped = anchor.min(n);
    if clamped > 0 && time_ms < times[clamped - 1] {
        lo = 0;
        hi = clamped;
    } else if clamped < n && time_ms >= times[clamped] {
        lo = clamped;
        hi = n;
    } else {
        return clamped;
    }
    while lo < hi {
        let mid = (lo + hi) / 2;
        if time_ms < times[mid] {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    lo
}

#[pyfunction]
fn keys_index_at(times: PyReadonlyArray1<'_, f64>, time_ms: f64, anchor: usize) -> PyResult<isize> {
    let times = times.as_slice()?;
    if times.is_empty() {
        return Ok(-1);
    }
    Ok(locate_right(times, time_ms, anchor) as isize - 1)
}

#[pyfunction]
fn interpolate_cursor_query(
    times: PyReadonlyArray1<'_, f64>,
    xs: PyReadonlyArray1<'_, f64>,
    ys: PyReadonlyArray1<'_, f64>,
    time_ms: f64,
    anchor: usize,
) -> PyResult<(usize, f64, f64)> {
    let times = times.as_slice()?;
    let xs = xs.as_slice()?;
    let ys = ys.as_slice()?;
    if times.is_empty() {
        return Ok((0, 256.0, 192.0));
    }
    let idx = locate_right(times, time_ms, anchor);
    if idx == 0 {
        return Ok((0, xs[0], ys[0]));
    }
    if idx >= times.len() {
        let last = times.len() - 1;
        return Ok((times.len(), xs[last], ys[last]));
    }
    let t0 = times[idx - 1];
    let t1 = times[idx];
    let dt = t1 - t0;
    if dt < 0.01 {
        return Ok((idx, xs[idx], ys[idx]));
    }
    let frac = (time_ms - t0) / dt;
    let x = xs[idx - 1] + (xs[idx] - xs[idx - 1]) * frac;
    let y = ys[idx - 1] + (ys[idx] - ys[idx - 1]) * frac;
    Ok((idx, x, y))
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn compute_slider_ball_instances(
    current_time_ms: f64,
    circle_radius: f32,
    ball_color: (f32, f32, f32),
    start_times: PyReadonlyArray1<'_, f32>,
    end_times: PyReadonlyArray1<'_, f32>,
    repeat_counts: PyReadonlyArray1<'_, i32>,
    path_starts: PyReadonlyArray1<'_, i32>,
    path_counts: PyReadonlyArray1<'_, i32>,
    path_points: PyReadonlyArray2<'_, f32>,
    z_values: PyReadonlyArray1<'_, f32>,
    mut out_buf: PyReadwriteArray2<'_, f32>,
    max_balls: usize,
) -> PyResult<usize> {
    let start_times = start_times.as_slice()?;
    let end_times = end_times.as_slice()?;
    let repeat_counts = repeat_counts.as_slice()?;
    let path_starts = path_starts.as_slice()?;
    let path_counts = path_counts.as_slice()?;
    let path_points = path_points.as_array();
    let z_values = z_values.as_slice()?;
    let mut out = out_buf.as_array_mut();
    let mut start_idx = 0usize;
    while start_idx < end_times.len() && f64::from(end_times[start_idx]) < current_time_ms {
        start_idx += 1;
    }
    let mut end_idx = 0usize;
    while end_idx < start_times.len() && f64::from(start_times[end_idx]) <= current_time_ms {
        end_idx += 1;
    }
    if end_idx <= start_idx {
        return Ok(0);
    }

    let (c0, c1, c2) = ball_color;
    let mut count = 0usize;
    for i in start_idx..end_idx {
        let start = f64::from(start_times[i]);
        let end = f64::from(end_times[i]);
        if current_time_ms < start || current_time_ms > end {
            continue;
        }
        let duration = end - start;
        if duration <= 0.0 {
            continue;
        }
        let repeats = repeat_counts[i] as i32;
        let progress = ((current_time_ms - start) / duration) as f32;
        let total_progress = progress * repeats as f32;
        let mut leg = total_progress.floor() as i32;
        let mut leg_t = total_progress - leg as f32;
        if leg >= repeats {
            leg = repeats - 1;
            leg_t = 1.0;
        }
        if leg % 2 == 1 {
            leg_t = 1.0 - leg_t;
        }
        let path_start = path_starts[i] as usize;
        let path_count = path_counts[i] as usize;
        if path_count < 2 {
            continue;
        }
        let float_idx = leg_t * (path_count as f32 - 1.0);
        let idx = (float_idx.floor() as usize).min(path_count - 2);
        let frac = float_idx - idx as f32;
        let p0 = path_points.row(path_start + idx);
        let p1 = path_points.row(path_start + idx + 1);
        out[[count, 0]] = p0[0] * (1.0 - frac) + p1[0] * frac;
        out[[count, 1]] = p0[1] * (1.0 - frac) + p1[1] * frac;
        out[[count, 2]] = circle_radius;
        out[[count, 3]] = c0;
        out[[count, 4]] = c1;
        out[[count, 5]] = c2;
        out[[count, 6]] = z_values[i] - 0.003;
        count += 1;
        if count >= max_balls {
            break;
        }
    }
    Ok(count)
}

#[pymodule]
fn _rust_core(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(keys_index_at, m)?)?;
    m.add_function(wrap_pyfunction!(interpolate_cursor_query, m)?)?;
    m.add_function(wrap_pyfunction!(compute_slider_ball_instances, m)?)?;
    Ok(())
}

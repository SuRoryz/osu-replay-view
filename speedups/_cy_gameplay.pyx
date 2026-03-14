# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True

"""Cython speedups for hot gameplay kernels."""

from cpython.tuple cimport PyTuple_New, PyTuple_SET_ITEM
from libc.math cimport fabs, sqrt
import numpy as np
cimport numpy as cnp


cpdef int keys_index_at(
    cnp.ndarray[cnp.float64_t, ndim=1] times,
    double time_ms,
    int anchor,
):
    cdef Py_ssize_t n = times.shape[0]
    cdef Py_ssize_t lo
    cdef Py_ssize_t hi
    cdef Py_ssize_t mid
    cdef Py_ssize_t idx

    if n == 0:
        return -1
    if anchor < 0:
        anchor = 0
    elif anchor > n:
        anchor = n

    if anchor > 0 and time_ms < times[anchor - 1]:
        lo = 0
        hi = anchor
    elif anchor < n and time_ms >= times[anchor]:
        lo = anchor
        hi = n
    else:
        return anchor - 1

    while lo < hi:
        mid = (lo + hi) >> 1
        if time_ms < times[mid]:
            hi = mid
        else:
            lo = mid + 1
    idx = lo - 1
    return <int>idx


cpdef tuple interpolate_cursor_query(
    cnp.ndarray[cnp.float64_t, ndim=1] times,
    cnp.ndarray[cnp.float64_t, ndim=1] xs,
    cnp.ndarray[cnp.float64_t, ndim=1] ys,
    double time_ms,
    int anchor,
):
    cdef Py_ssize_t n = times.shape[0]
    cdef Py_ssize_t lo
    cdef Py_ssize_t hi
    cdef Py_ssize_t mid
    cdef Py_ssize_t idx
    cdef double t0
    cdef double t1
    cdef double dt
    cdef double frac
    cdef double x
    cdef double y
    cdef tuple result

    if n == 0:
        return (0, 256.0, 192.0)
    if anchor < 0:
        anchor = 0
    elif anchor > n:
        anchor = n

    if anchor > 0 and time_ms < times[anchor - 1]:
        lo = 0
        hi = anchor
    elif anchor < n and time_ms >= times[anchor]:
        lo = anchor
        hi = n
    else:
        lo = anchor
        hi = anchor

    if lo != hi:
        while lo < hi:
            mid = (lo + hi) >> 1
            if time_ms < times[mid]:
                hi = mid
            else:
                lo = mid + 1
        idx = lo
    else:
        idx = lo

    if idx <= 0:
        return (0, float(xs[0]), float(ys[0]))
    if idx >= n:
        return (n, float(xs[n - 1]), float(ys[n - 1]))

    t0 = times[idx - 1]
    t1 = times[idx]
    dt = t1 - t0
    if dt < 0.01:
        return (idx, float(xs[idx]), float(ys[idx]))
    frac = (time_ms - t0) / dt
    x = xs[idx - 1] + (xs[idx] - xs[idx - 1]) * frac
    y = ys[idx - 1] + (ys[idx] - ys[idx - 1]) * frac
    return (idx, x, y)


cpdef int compute_slider_ball_instances(
    double current_time_ms,
    float circle_radius,
    object ball_color,
    cnp.ndarray[cnp.float32_t, ndim=1] start_times,
    cnp.ndarray[cnp.float32_t, ndim=1] end_times,
    cnp.ndarray[cnp.int32_t, ndim=1] repeat_counts,
    cnp.ndarray[cnp.int32_t, ndim=1] path_starts,
    cnp.ndarray[cnp.int32_t, ndim=1] path_counts,
    cnp.ndarray[cnp.float32_t, ndim=2] path_points,
    cnp.ndarray[cnp.float32_t, ndim=1] z_values,
    cnp.ndarray[cnp.float32_t, ndim=2] out_buf,
    int max_balls,
):
    cdef Py_ssize_t n = start_times.shape[0]
    cdef Py_ssize_t i
    cdef int start_idx = 0
    cdef int end_idx = 0
    cdef int ball_count = 0
    cdef float start
    cdef float end
    cdef float duration
    cdef int repeats
    cdef float progress
    cdef float total_progress
    cdef int leg
    cdef float leg_t
    cdef int path_start
    cdef int path_count
    cdef float float_idx
    cdef int idx
    cdef float frac
    cdef cnp.float32_t c0 = <cnp.float32_t>ball_color[0]
    cdef cnp.float32_t c1 = <cnp.float32_t>ball_color[1]
    cdef cnp.float32_t c2 = <cnp.float32_t>ball_color[2]

    while start_idx < n and end_times[start_idx] < current_time_ms:
        start_idx += 1
    while end_idx < n and start_times[end_idx] <= current_time_ms:
        end_idx += 1
    if end_idx <= start_idx:
        return 0

    for i in range(start_idx, end_idx):
        start = start_times[i]
        end = end_times[i]
        if current_time_ms < start or current_time_ms > end:
            continue
        duration = end - start
        if duration <= 0.0:
            continue

        repeats = repeat_counts[i]
        progress = <float>((current_time_ms - start) / duration)
        total_progress = progress * repeats
        leg = <int>total_progress
        leg_t = total_progress - leg
        if leg >= repeats:
            leg = repeats - 1
            leg_t = 1.0
        if leg & 1:
            leg_t = 1.0 - leg_t

        path_start = path_starts[i]
        path_count = path_counts[i]
        if path_count < 2:
            continue
        float_idx = leg_t * (path_count - 1)
        idx = <int>float_idx
        if idx > path_count - 2:
            idx = path_count - 2
        frac = float_idx - idx

        out_buf[ball_count, 0] = path_points[path_start + idx, 0] * (1.0 - frac) + path_points[path_start + idx + 1, 0] * frac
        out_buf[ball_count, 1] = path_points[path_start + idx, 1] * (1.0 - frac) + path_points[path_start + idx + 1, 1] * frac
        out_buf[ball_count, 2] = circle_radius
        out_buf[ball_count, 3] = c0
        out_buf[ball_count, 4] = c1
        out_buf[ball_count, 5] = c2
        out_buf[ball_count, 6] = z_values[i] - 0.003
        ball_count += 1
        if ball_count >= max_balls:
            break

    return ball_count


cdef void _interp_trail_xy(
    cnp.ndarray[cnp.float32_t, ndim=2] raw_buf,
    int count,
    double target_time,
    float* out_x,
    float* out_y,
):
    cdef int hi = count - 1
    cdef int lo = 0
    cdef int mid
    cdef int idx
    cdef double t0
    cdef double t1
    cdef double frac
    if count <= 0:
        out_x[0] = 0.0
        out_y[0] = 0.0
        return
    if target_time <= raw_buf[0, 0]:
        out_x[0] = raw_buf[0, 1]
        out_y[0] = raw_buf[0, 2]
        return
    if target_time >= raw_buf[count - 1, 0]:
        out_x[0] = raw_buf[count - 1, 1]
        out_y[0] = raw_buf[count - 1, 2]
        return
    while lo + 1 < hi:
        mid = (lo + hi) >> 1
        if raw_buf[mid, 0] <= target_time:
            lo = mid
        else:
            hi = mid
    idx = lo
    t0 = raw_buf[idx, 0]
    t1 = raw_buf[idx + 1, 0]
    if t1 - t0 <= 1e-6:
        out_x[0] = raw_buf[idx + 1, 1]
        out_y[0] = raw_buf[idx + 1, 2]
        return
    frac = (target_time - t0) / (t1 - t0)
    out_x[0] = <float>(raw_buf[idx, 1] + (raw_buf[idx + 1, 1] - raw_buf[idx, 1]) * frac)
    out_y[0] = <float>(raw_buf[idx, 2] + (raw_buf[idx + 1, 2] - raw_buf[idx, 2]) * frac)


cpdef tuple prepare_trail_control_points(
    object trail_points,
    int max_points,
    double current_time_ms,
    double trail_lifetime_ms,
    cnp.ndarray[cnp.float32_t, ndim=2] raw_buf,
    cnp.ndarray[cnp.float32_t, ndim=2] out_buf,
):
    cdef int src_count = len(trail_points)
    cdef int src_start = 0
    cdef int count
    cdef int i
    cdef int clip_count = 0
    cdef object point
    cdef double start_time
    cdef double end_time = current_time_ms
    cdef float x
    cdef float y
    cdef double t
    cdef double dx
    cdef double dy
    cdef double total_len = 0.0

    if src_count < 2:
        return 0, 0.0
    if src_count > max_points:
        src_start = src_count - max_points
    count = src_count - src_start
    for i in range(count):
        point = trail_points[src_start + i]
        raw_buf[i, 0] = <float>point[0]
        raw_buf[i, 1] = <float>point[1]
        raw_buf[i, 2] = <float>point[2]

    start_time = raw_buf[0, 0]
    if current_time_ms - trail_lifetime_ms > start_time:
        start_time = current_time_ms - trail_lifetime_ms
    if end_time - start_time <= 1.0:
        return 0, 0.0

    _interp_trail_xy(raw_buf, count, start_time, &x, &y)
    out_buf[clip_count, 0] = x
    out_buf[clip_count, 1] = y
    out_buf[clip_count, 2] = <float>start_time
    out_buf[clip_count, 3] = 0.0
    clip_count += 1

    for i in range(count):
        t = raw_buf[i, 0]
        if start_time < t < end_time:
            out_buf[clip_count, 0] = raw_buf[i, 1]
            out_buf[clip_count, 1] = raw_buf[i, 2]
            out_buf[clip_count, 2] = raw_buf[i, 0]
            out_buf[clip_count, 3] = 0.0
            clip_count += 1

    _interp_trail_xy(raw_buf, count, end_time, &x, &y)
    if (
        clip_count == 0
        or fabs(x - out_buf[clip_count - 1, 0]) > 1e-4
        or fabs(y - out_buf[clip_count - 1, 1]) > 1e-4
    ):
        out_buf[clip_count, 0] = x
        out_buf[clip_count, 1] = y
        out_buf[clip_count, 2] = <float>end_time
        out_buf[clip_count, 3] = 0.0
        clip_count += 1

    if clip_count < 2:
        return 0, 0.0

    for i in range(1, clip_count):
        dx = out_buf[i, 0] - out_buf[i - 1, 0]
        dy = out_buf[i, 1] - out_buf[i - 1, 1]
        total_len += sqrt(dx * dx + dy * dy)
    return clip_count, total_len


cpdef int clip_trail_points(
    cnp.ndarray[cnp.float32_t, ndim=2] raw_buf,
    int count,
    double current_time_ms,
    double trail_lifetime_ms,
    double min_control_dt_ms,
    double min_control_dist_sq,
    cnp.ndarray[cnp.float32_t, ndim=2] clip_buf,
):
    cdef int i
    cdef int clip_count = 0
    cdef int write_idx
    cdef double start_time
    cdef double end_time = current_time_ms
    cdef double t
    cdef double dt
    cdef double dx
    cdef double dy
    cdef double last_t
    cdef double last_x
    cdef double last_y
    cdef float start_x
    cdef float start_y
    cdef float end_x
    cdef float end_y

    if count < 2:
        return 0

    start_time = raw_buf[0, 0]
    if current_time_ms - trail_lifetime_ms > start_time:
        start_time = current_time_ms - trail_lifetime_ms
    if end_time - start_time <= 1.0:
        return 0

    _interp_trail_xy(raw_buf, count, start_time, &start_x, &start_y)
    clip_buf[clip_count, 0] = <cnp.float32_t>start_time
    clip_buf[clip_count, 1] = start_x
    clip_buf[clip_count, 2] = start_y
    clip_count += 1

    for i in range(count):
        t = raw_buf[i, 0]
        if start_time < t < end_time:
            clip_buf[clip_count, 0] = raw_buf[i, 0]
            clip_buf[clip_count, 1] = raw_buf[i, 1]
            clip_buf[clip_count, 2] = raw_buf[i, 2]
            clip_count += 1

    _interp_trail_xy(raw_buf, count, end_time, &end_x, &end_y)
    if (
        clip_count == 0
        or fabs(end_x - clip_buf[clip_count - 1, 1]) > 1e-4
        or fabs(end_y - clip_buf[clip_count - 1, 2]) > 1e-4
    ):
        clip_buf[clip_count, 0] = <cnp.float32_t>end_time
        clip_buf[clip_count, 1] = end_x
        clip_buf[clip_count, 2] = end_y
        clip_count += 1

    if clip_count < 2:
        return clip_count

    write_idx = 1
    last_t = clip_buf[0, 0]
    last_x = clip_buf[0, 1]
    last_y = clip_buf[0, 2]
    for i in range(1, clip_count):
        t = clip_buf[i, 0]
        dx = clip_buf[i, 1] - last_x
        dy = clip_buf[i, 2] - last_y
        dt = t - last_t
        if dt <= min_control_dt_ms or dx * dx + dy * dy <= min_control_dist_sq:
            clip_buf[write_idx - 1, 0] = clip_buf[i, 0]
            clip_buf[write_idx - 1, 1] = clip_buf[i, 1]
            clip_buf[write_idx - 1, 2] = clip_buf[i, 2]
        else:
            clip_buf[write_idx, 0] = clip_buf[i, 0]
            clip_buf[write_idx, 1] = clip_buf[i, 1]
            clip_buf[write_idx, 2] = clip_buf[i, 2]
            write_idx += 1
        last_t = clip_buf[write_idx - 1, 0]
        last_x = clip_buf[write_idx - 1, 1]
        last_y = clip_buf[write_idx - 1, 2]

    return write_idx


cdef void _interp_trail_xy_ring(
    cnp.ndarray[cnp.float64_t, ndim=2] trail_buf,
    int start_idx,
    int count,
    double target_time,
    float* out_x,
    float* out_y,
):
    cdef int max_points = trail_buf.shape[0]
    cdef int lo = 0
    cdef int hi = count - 1
    cdef int mid
    cdef int mid_idx
    cdef int next_idx
    cdef int ring_idx
    cdef double t0
    cdef double t1
    cdef double frac
    if count <= 0:
        out_x[0] = 0.0
        out_y[0] = 0.0
        return
    if target_time <= trail_buf[start_idx, 0]:
        out_x[0] = trail_buf[start_idx, 1]
        out_y[0] = trail_buf[start_idx, 2]
        return
    ring_idx = (start_idx + count - 1) % max_points
    if target_time >= trail_buf[ring_idx, 0]:
        out_x[0] = trail_buf[ring_idx, 1]
        out_y[0] = trail_buf[ring_idx, 2]
        return
    while lo + 1 < hi:
        mid = (lo + hi) >> 1
        mid_idx = (start_idx + mid) % max_points
        if trail_buf[mid_idx, 0] <= target_time:
            lo = mid
        else:
            hi = mid
    ring_idx = (start_idx + lo) % max_points
    next_idx = (ring_idx + 1) % max_points
    t0 = trail_buf[ring_idx, 0]
    t1 = trail_buf[next_idx, 0]
    if t1 - t0 <= 1e-6:
        out_x[0] = trail_buf[next_idx, 1]
        out_y[0] = trail_buf[next_idx, 2]
        return
    frac = (target_time - t0) / (t1 - t0)
    out_x[0] = <float>(trail_buf[ring_idx, 1] + (trail_buf[next_idx, 1] - trail_buf[ring_idx, 1]) * frac)
    out_y[0] = <float>(trail_buf[ring_idx, 2] + (trail_buf[next_idx, 2] - trail_buf[ring_idx, 2]) * frac)


cpdef tuple advance_trail_state(
    cnp.ndarray[cnp.float64_t, ndim=2] trail_buf,
    int start_idx,
    int count,
    double current_time_ms,
    double x,
    double y,
    double trail_lifetime_ms,
    double min_sample_dt_ms,
    double min_sample_dist_sq,
):
    cdef int max_points = trail_buf.shape[0]
    cdef int anchor_idx
    cdef int last_idx
    cdef int write_idx
    cdef double dt
    cdef double dx
    cdef double dy

    while count > 0 and current_time_ms - trail_buf[start_idx, 0] > trail_lifetime_ms:
        start_idx += 1
        if start_idx >= max_points:
            start_idx = 0
        count -= 1

    if count <= 0:
        trail_buf[0, 0] = current_time_ms
        trail_buf[0, 1] = x
        trail_buf[0, 2] = y
        return 0, 1

    if count == 1:
        write_idx = (start_idx + 1) % max_points
        trail_buf[write_idx, 0] = current_time_ms
        trail_buf[write_idx, 1] = x
        trail_buf[write_idx, 2] = y
        return start_idx, 2

    anchor_idx = (start_idx + count - 2) % max_points
    dt = current_time_ms - trail_buf[anchor_idx, 0]
    dx = x - trail_buf[anchor_idx, 1]
    dy = y - trail_buf[anchor_idx, 2]

    if dt >= min_sample_dt_ms or dx * dx + dy * dy >= min_sample_dist_sq:
        if count >= max_points:
            start_idx += 1
            if start_idx >= max_points:
                start_idx = 0
            count -= 1
        write_idx = (start_idx + count) % max_points
        trail_buf[write_idx, 0] = current_time_ms
        trail_buf[write_idx, 1] = x
        trail_buf[write_idx, 2] = y
        count += 1
    else:
        last_idx = (start_idx + count - 1) % max_points
        trail_buf[last_idx, 0] = current_time_ms
        trail_buf[last_idx, 1] = x
        trail_buf[last_idx, 2] = y

    return start_idx, count


cpdef tuple prepare_trail_control_points_ring(
    cnp.ndarray[cnp.float64_t, ndim=2] trail_buf,
    int start_idx,
    int count,
    double current_time_ms,
    double trail_lifetime_ms,
    cnp.ndarray[cnp.float32_t, ndim=2] out_buf,
):
    cdef int max_points = trail_buf.shape[0]
    cdef int i
    cdef int ring_idx
    cdef int clip_count = 0
    cdef double start_time
    cdef double end_time = current_time_ms
    cdef double t
    cdef double dx
    cdef double dy
    cdef cnp.float32_t seg_len
    cdef cnp.float32_t total_len = 0.0
    cdef float x
    cdef float y

    if count < 2:
        return 0, 0.0

    start_time = trail_buf[start_idx, 0]
    if current_time_ms - trail_lifetime_ms > start_time:
        start_time = current_time_ms - trail_lifetime_ms
    if end_time - start_time <= 1.0:
        return 0, 0.0

    _interp_trail_xy_ring(trail_buf, start_idx, count, start_time, &x, &y)
    out_buf[clip_count, 0] = x
    out_buf[clip_count, 1] = y
    out_buf[clip_count, 2] = <float>start_time
    out_buf[clip_count, 3] = 0.0
    clip_count += 1

    for i in range(count):
        ring_idx = (start_idx + i) % max_points
        t = trail_buf[ring_idx, 0]
        if start_time < t < end_time:
            out_buf[clip_count, 0] = trail_buf[ring_idx, 1]
            out_buf[clip_count, 1] = trail_buf[ring_idx, 2]
            out_buf[clip_count, 2] = trail_buf[ring_idx, 0]
            out_buf[clip_count, 3] = 0.0
            clip_count += 1

    _interp_trail_xy_ring(trail_buf, start_idx, count, end_time, &x, &y)
    if (
        fabs(x - out_buf[clip_count - 1, 0]) > 1e-4
        or fabs(y - out_buf[clip_count - 1, 1]) > 1e-4
    ):
        out_buf[clip_count, 0] = x
        out_buf[clip_count, 1] = y
        out_buf[clip_count, 2] = <float>end_time
        out_buf[clip_count, 3] = 0.0
        clip_count += 1

    if clip_count < 2:
        return 0, 0.0

    for i in range(1, clip_count):
        dx = out_buf[i, 0] - out_buf[i - 1, 0]
        dy = out_buf[i, 1] - out_buf[i - 1, 1]
        seg_len = <cnp.float32_t>sqrt(dx * dx + dy * dy)
        total_len += seg_len
    return clip_count, float(total_len)

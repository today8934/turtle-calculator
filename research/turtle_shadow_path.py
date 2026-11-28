"""Track the unpyramided hypothetical breakout used by the System 1 skip rule."""

import numpy as np

from kospi_system1 import TICK_REFORM, etf_flags, executable


EPS = 1e-7


def crossed(start, finish, level, sign, opening):
    """A gap trades only at the open. A continuous move must newly reach the line."""
    return (sign * (finish - level) >= -EPS
            and (opening or sign * (start - level) < -EPS))


def shadow_events(m, dates, phases, return_short=False):
    """Return eligible entry levels, in adjusted price units, by date, phase, ticker.

    The caller models cash, actual positions and fills. Gapped orders fill at the
    open even though the returned value identifies the threshold they crossed.
    With return_short=True, the result is a pair of long and short arrays.
    """
    days, codes = m["o"].shape
    if phases[0] != "o" or len(dates) != days:
        raise ValueError("the first phase must be the open of each supplied date")
    long_orders = np.full((days, len(phases), codes), np.nan)
    short_orders = np.full_like(long_orders, np.nan) if return_short else None
    old = np.asarray(dates < TICK_REFORM)
    etfs = etf_flags(m)
    path = [m[name] for name in phases]
    for k in range(codes):
        side, last_won = 0, False
        entry = raw_stop = stop = stop_factor = np.nan
        stop_old = False
        available = np.flatnonzero((m["n"][:,k] > 0) & np.isfinite(m["u20"][:,k]))
        for i in available:
            n, factor = m["n"][i,k], m["factor"][i,k]
            if side and (factor != stop_factor or old[i] != stop_old):
                stop = executable(raw_stop, factor, side < 0, old[i], etfs[k])
                stop_factor, stop_old = factor, old[i]
            up, down = m["u20"][i,k], m["d20"][i,k]
            up55 = m["u55"][i,k]
            down55 = m["d55"][i,k] if return_short else np.nan
            lower, upper = m["d10"][i,k], m["u10"][i,k]
            start = np.nan
            for phase, leg in enumerate(path):
                finish = leg[i,k]
                opening = phase == 0
                if not opening and abs(finish - start) <= EPS:
                    start = finish
                    continue
                before = last_won
                resolved = np.nan
                if side:
                    channel = lower if side > 0 else upper
                    if crossed(start, finish, channel, -side, opening):
                        channel_fill = finish if opening else channel
                        if side * (channel_fill - entry) > EPS:
                            resolved, last_won = channel_fill, True
                    if np.isnan(resolved) and crossed(start, finish, stop, -side, opening):
                        resolved, last_won = (finish if opening else stop), False
                    if np.isfinite(resolved):
                        side = 0

                # Only the open can cross in either direction without a preceding
                # intraday price. The daily up/down entry channels never overlap.
                directions = (1, -1) if opening else ((1,) if finish > start else (-1,))
                for sign in directions:
                    near, far = (up, up55) if sign > 0 else (down, down55)
                    near_crossed = crossed(start, finish, near, sign, opening)
                    after_exit = (np.isfinite(resolved) and
                                  (opening or sign * (near - resolved) >= -EPS))
                    filter_at_near = last_won if after_exit else before
                    if near_crossed and side == 0 and (np.isnan(resolved) or after_exit):
                        # Opening at a new record after an earlier record does not
                        # reset an outstanding paper trade or its original 2N stop.
                        side = sign
                        entry = finish if opening else near
                        raw_stop = entry - sign * 2 * n
                        stop = executable(raw_stop, factor, sign < 0, old[i], etfs[k])
                        stop_factor, stop_old = factor, old[i]

                    orders = long_orders if sign > 0 else short_orders
                    if orders is None:
                        continue
                    if near_crossed and not filter_at_near:
                        orders[i,phase,k] = near
                    elif crossed(start, finish, far, sign, opening):
                        after_far = (np.isfinite(resolved) and
                                     (opening or sign * (far - resolved) >= -EPS))
                        filter_at_far = last_won if after_far else before
                        if filter_at_far:
                            orders[i,phase,k] = far
                start = finish
    return (long_orders, short_orders) if return_short else long_orders

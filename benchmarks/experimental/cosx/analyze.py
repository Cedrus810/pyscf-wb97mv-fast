#!/usr/bin/env python
"""Summarize result_*.json from sgx_locality.py into N_SR / N_LR tables.

Decision rule (ESP-integral part of K only):
    R_N = N_SR / N_LR       work ratio, counted in ESP integrals
    R_t = t_SR / t_LR       cost per ESP integral
    R_T ~= R_N * R_t
R_N from the oracle is what an ideal screening + dedicated kernel could reach;
R_N from stock is what PySCF SGX gets today.
"""
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RN_WORTH = 0.5      # R_N^oracle at or below this: dedicated SR kernel is worth it
RN_NOT_WORTH = 0.75  # at or above this: full+SR is hard to justify


def verdict(rn):
    if rn <= RN_WORTH:
        return 'dedicated SR kernel worth it'
    if rn >= RN_NOT_WORTH:
        return 'full+SR hard to justify'
    return 'gray zone'


def rule_summary(info, st, orc):
    s = {k: st[f'{k}|sample_pos|dmscr'] for k in ('full', 'LR', 'SR')}
    cost = {k: s[k]['t_int_thread_s'] / s[k]['ints'] for k in s}
    rn_stock = s['SR']['ints'] / s['LR']['ints']
    rt = cost['SR'] / cost['LR']
    rt_full = cost['full'] / cost['LR']
    print('\n[decision rule]  R_T ~= R_N * R_t   (ESP integrals only)')
    print(f"  per-int cost (ns): full {cost['full']*1e9:.1f}  LR {cost['LR']*1e9:.1f}  "
          f"SR {cost['SR']*1e9:.1f}   R_t = {rt:.2f}  (SR at full cost -> R_t = {rt_full:.2f})")
    print(f"  stock : R_N = {rn_stock:.3f}   R_N*R_t = {rn_stock*rt:.3f}   "
          f"measured t_int ratio = {s['SR']['t_int_thread_s']/s['LR']['t_int_thread_s']:.3f}   "
          f"wall ratio = {s['SR']['wall_s']/s['LR']['wall_s']:.3f}")
    if not orc:
        return
    o = {k: orc[k] for k in ('full', 'LR', 'SR')}
    rows = [('E ' + b, o['SR']['energy'][b][1] / o['LR']['energy'][b][1])
            for b in ('1e-13', '1e-10', '1e-08')]
    rows += [('V/blk ' + b, o['SR']['potential'][b][1] / o['LR']['potential'][b][1])
             for b in ('1e-06', '1e-07')]
    print(f"  {'oracle budget':14s} {'R_N':>6s} {'R_T(stock R_t)':>15s} "
          f"{'R_T(R_t=1)':>11s} {'R_T(SR=full)':>13s}  verdict")
    for name, rn in rows:
        print(f"  {name:14s} {rn:6.3f} {rn*rt:15.3f} {rn:11.3f} {rn*rt_full:13.3f}  {verdict(rn)}")


def main(paths):
    for path in paths:
        r = json.load(open(path))
        info, st, orc = r['info'], r['stock'], r['oracle']
        print(f"\n=== {info['system']} / {info['basis']}: nao={info['nao']} nbas={info['nbas']} "
              f"ngrids={info['ngrids']} blocks={info['nblk']} pair_mask_pairs={info['pair_mask_pairs']} "
              f"threads={info['threads']}")

        print('\n[stock SGX, per K build]')
        print(f"{'config':26s} {'tasks(M)':>9s} {'tasks/blk':>9s} {'ints(G)':>8s} "
              f"{'ns/int':>7s} {'t_int/wall':>10s} {'wall(s)':>8s}")
        for key, c in st.items():
            nsint = c['t_int_thread_s'] / c['ints'] * 1e9
            frac = c['t_int_thread_s'] / info['threads'] / c['wall_s']
            print(f"{key:26s} {c['calls']/1e6:9.2f} {c['tasks_per_block_mean']:9.0f} "
                  f"{c['ints']/1e9:8.2f} {nsint:7.1f} {frac:10.2f} {c['wall_s']:8.2f}")
        cost = {k: st[f'{k}|sample_pos|dmscr']['t_int_thread_s'] / st[f'{k}|sample_pos|dmscr']['ints']
                for k in ('full', 'LR', 'SR')}
        s = {k: st[f'{k}|sample_pos|dmscr'] for k in ('full', 'LR', 'SR')}
        print(f"stock N_SR/N_LR: tasks {s['SR']['calls']/s['LR']['calls']:.3f}  "
              f"ints {s['SR']['ints']/s['LR']['ints']:.3f}  "
              f"wall T_SR/T_LR {s['SR']['wall_s']/s['LR']['wall_s']:.3f}  "
              f"per-int cost SR/LR {cost['SR']/cost['LR']:.2f}, SR/full {cost['SR']/cost['full']:.2f}")
        if r.get('accuracy'):
            a = r['accuracy']
            print('stock error vs no-DM-screening: ' + '  '.join(
                f"{k}: dE={v['dE']:.1e} dKmax={v['dK_max']:.1e}" for k, v in a.items()))

        rule_summary(info, st, orc)
        if not orc:
            continue
        o = {k: orc[k] for k in ('full', 'LR', 'SR')}
        print(f"\n[oracle on {o['full']['sample_blocks']} sampled blocks; "
              f"dense tasks {o['full']['dense_tasks']}]")
        print(f"{'':14s} {'full':>9s} {'LR':>9s} {'SR':>9s} {'SR/LR':>7s}")
        print(f"{'stock':14s} " + ' '.join(f"{o[k]['stock_tasks']:9d}" for k in o)
              + f" {o['SR']['stock_tasks']/o['LR']['stock_tasks']:7.3f}")
        for b in o['full']['energy']:
            n = [o[k]['energy'][b][0] for k in o]
            print(f"{'E budget '+b:14s} " + ' '.join(f"{x:9d}" for x in n) + f" {n[2]/n[1]:7.3f}")
        for b in o['full']['potential']:
            n = [o[k]['potential'][b][0] for k in o]
            print(f"{'V/blk '+b:14s} " + ' '.join(f"{x:9d}" for x in n) + f" {n[2]/n[1]:7.3f}")

        # decomposition totals, integral-count weighted, costed with measured ns/int
        print('\n[decomposition at total energy budget eps, oracle screening]')
        print(f"{'eps':>7s} {'A tasks':>9s} {'B tasks':>9s} {'B/A':>6s} "
              f"{'B/A time(stock kernel)':>23s} {'B/A time(SR kernel=full cost)':>30s}")
        stock_a = (o['full']['stock_ints'] * cost['full'] + o['LR']['stock_ints'] * cost['LR'])
        for eps, d in orc['decomposition'].items():
            ta = d['A_full'][1] * cost['full'] + d['A_LR'][1] * cost['LR']
            tb = d['B_full'][1] * cost['full'] + d['B_SR'][1] * cost['SR']
            tb2 = d['B_full'][1] * cost['full'] + d['B_SR'][1] * cost['full']
            print(f"{eps:>7s} {d['A_total'][0]:9d} {d['B_total'][0]:9d} "
                  f"{d['B_total'][0]/d['A_total'][0]:6.3f} {tb/ta:23.3f} {tb2/ta:30.3f}"
                  f"   (A_oracle/A_stock time {ta/stock_a:.3f})")


if __name__ == '__main__':
    main(sys.argv[1:] or sorted(glob.glob(os.path.join(HERE, 'result_*.json'))))

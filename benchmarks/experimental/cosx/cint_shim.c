/*
 * Counting shim for libcint int1e_grids_{sph,cart}.
 *
 * PySCF's SGXnr_direct_k_drv receives the ESP integral function as a pointer
 * from Python, so we can hand it this wrapper instead. Every call that
 * survives SGX's C-level screening lands here as one (ish, jsh, grid range)
 * task; we count it and forward to the real libcint function.
 *
 * Build: gcc -O2 -fPIC -shared -o libcint_shim.so cint_shim.c
 */
#include <stdint.h>
#include <string.h>
#include <time.h>

#define BAS_SLOTS 8
#define ANG_OF    1
#define NCTR_OF   3
#define LMAX      8

typedef int (*cintor_t)(double *out, int *dims, int *shls, int *atm, int natm,
                        int *bas, int nbas, double *env, void *opt,
                        double *cache);

static cintor_t target = NULL;
static int cart = 0;
static int grid_offset = 0;

static long long n_calls;       /* evaluated (shell pair, grid range) tasks */
static long long n_points;      /* sum of grid points over tasks */
static long long n_ints;        /* ESP integrals = points * nfi * nfj */
static long long t_ns;          /* thread-summed time inside libcint */
static long long ints_by_l[LMAX][LMAX];

static int *rec = NULL;         /* optional per-task record: ish, jsh, g0 */
static long long rec_cap = 0;
static long long rec_n = 0;

void shim_set_target(void *f, int is_cart) { target = (cintor_t)f; cart = is_cart; }
void shim_set_offset(int off) { grid_offset = off; }
void shim_set_record(int *buf, long long cap) { rec = buf; rec_cap = cap; rec_n = 0; }

void shim_reset(void)
{
    n_calls = n_points = n_ints = t_ns = 0;
    rec_n = 0;
    memset(ints_by_l, 0, sizeof(ints_by_l));
}

/* out[0..4] = calls, points, ints, t_ns, rec_n ; out[5..] = ints_by_l */
void shim_get(long long *out)
{
    out[0] = n_calls; out[1] = n_points; out[2] = n_ints; out[3] = t_ns;
    out[4] = rec_n;
    memcpy(out + 5, ints_by_l, sizeof(ints_by_l));
}

static inline long long nfunc(int l, int nctr)
{
    return (cart ? (l + 1) * (l + 2) / 2 : 2 * l + 1) * (long long)nctr;
}

int shim_int1e_grids(double *out, int *dims, int *shls, int *atm, int natm,
                     int *bas, int nbas, double *env, void *opt, double *cache)
{
    if (out == NULL) {  /* cache-size query */
        return target(out, dims, shls, atm, natm, bas, nbas, env, opt, cache);
    }
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    int r = target(out, dims, shls, atm, natm, bas, nbas, env, opt, cache);
    clock_gettime(CLOCK_MONOTONIC, &t1);

    int ish = shls[0], jsh = shls[1];
    int li = bas[ish * BAS_SLOTS + ANG_OF], lj = bas[jsh * BAS_SLOTS + ANG_OF];
    long long ng = shls[3] - shls[2];
    long long ni = ng * nfunc(li, bas[ish * BAS_SLOTS + NCTR_OF])
                      * nfunc(lj, bas[jsh * BAS_SLOTS + NCTR_OF]);
    long long dt = (t1.tv_sec - t0.tv_sec) * 1000000000LL + (t1.tv_nsec - t0.tv_nsec);

    __atomic_add_fetch(&n_calls, 1, __ATOMIC_RELAXED);
    __atomic_add_fetch(&n_points, ng, __ATOMIC_RELAXED);
    __atomic_add_fetch(&n_ints, ni, __ATOMIC_RELAXED);
    __atomic_add_fetch(&t_ns, dt, __ATOMIC_RELAXED);
    if (li < LMAX && lj < LMAX) {
        __atomic_add_fetch(&ints_by_l[li][lj], ni, __ATOMIC_RELAXED);
    }
    if (rec) {
        long long k = __atomic_fetch_add(&rec_n, 1, __ATOMIC_RELAXED);
        if (k < rec_cap) {
            rec[3 * k] = ish;
            rec[3 * k + 1] = jsh;
            rec[3 * k + 2] = shls[2] + grid_offset;
        }
    }
    return r;
}

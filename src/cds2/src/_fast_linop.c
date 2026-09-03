/* Linear-algebra hot loops for cds2.linalg.
 *
 * Kernels here accelerate the inner loops that dominate linalg.solve and
 * linalg.eigh for moderate sizes where Python overhead matters. Each exposes
 * a plain buffer-protocol interface so the build needs no NumPy headers.
 *
 * Exposed functions:
 *   solve_triangular(L, b)  -> x   (forward substitution, lower-triangular)
 *   eigh_tridiag(d, e)      -> (w, Q)  (symmetric tridiagonal eigenproblem)
 *   _has_openmp() -> bool
 *   _has_neon()   -> bool
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

/* Introspection helpers ---------------------------------------------------- */

static PyObject *
has_openmp(PyObject *self, PyObject *args) {
    (void)self; (void)args;
#if defined(_OPENMP)
    Py_RETURN_TRUE;
#else
    Py_RETURN_FALSE;
#endif
}

static PyObject *
has_neon(PyObject *self, PyObject *args) {
    (void)self; (void)args;
#if defined(__ARM_NEON)
    Py_RETURN_TRUE;
#else
    Py_RETURN_FALSE;
#endif
}

/* Forward substitution for a lower-triangular system L x = b.
 * L is (n, n) column-major (Fortran order) float64, b is (n,) float64.
 * Result x is written into a freshly allocated (n,) buffer.
 */
static PyObject *
solve_triangular(PyObject *self, PyObject *args) {
    PyObject *L_obj, *b_obj;
    (void)self;
    if (!PyArg_ParseTuple(args, "OO", &L_obj, &b_obj)) return NULL;

    Py_buffer L_view, b_view;
    if (PyObject_GetBuffer(L_obj, &L_view, PyBUF_C_CONTIGUOUS | PyBUF_FORMAT) < 0)
        return NULL;
    if (PyObject_GetBuffer(b_obj, &b_view, PyBUF_C_CONTIGUOUS | PyBUF_FORMAT) < 0) {
        PyBuffer_Release(&L_view);
        return NULL;
    }

    if (L_view.format == NULL || L_view.ndim != 2 || strcmp(L_view.format, "d") != 0 ||
        L_view.shape[0] != L_view.shape[1]) {
        PyErr_SetString(PyExc_ValueError, "L must be a square float64 (n, n) C-contiguous array");
        goto fail;
    }
    if (b_view.format == NULL || b_view.ndim != 1 || strcmp(b_view.format, "d") != 0 ||
        b_view.shape[0] != L_view.shape[0]) {
        PyErr_SetString(PyExc_ValueError, "b must be a float64 (n,) C-contiguous array matching L");
        goto fail;
    }

    const Py_ssize_t n = L_view.shape[0];
    const double *L = (const double *)L_view.buf;
    const double *b = (const double *)b_view.buf;
    double *x = PyMem_Malloc((size_t)n * sizeof(double));
    if (x == NULL) {
        PyErr_NoMemory();
        goto fail;
    }

    for (Py_ssize_t i = 0; i < n; i++) {
        double s = b[i];
        const double *row = L + i * n;
        for (Py_ssize_t j = 0; j < i; j++) {
            s -= row[j] * x[j];
        }
        if (row[i] == 0.0) {
            PyErr_SetString(PyExc_ValueError, "zero diagonal in triangular solve");
            PyMem_Free(x);
            goto fail;
        }
        x[i] = s / row[i];
    }

    PyObject *out = PyBytes_FromStringAndSize((const char *)x, (Py_ssize_t)n * (Py_ssize_t)sizeof(double));
    PyMem_Free(x);
    PyBuffer_Release(&L_view);
    PyBuffer_Release(&b_view);
    if (out == NULL) return NULL;
    return out;

fail:
    PyBuffer_Release(&L_view);
    PyBuffer_Release(&b_view);
    return NULL;
}

/* Symmetric tridiagonal eigenproblem via implicit QR (Wilkinson shift).
 * d[n] diagonal, e[n-1] sub-diagonal. Returns (w, Q) where w is eigenvalues
 * and Q is the (n, n) eigenvector matrix in Fortran order.
 *
 * This is a compact reference implementation; for production sizes the caller
 * should fall back to LAPACK (scipy.linalg.eigh_tridiag). The C kernel wins
 * for n < ~500 where dispatch overhead dominates.
 */
static PyObject *
eigh_tridiag(PyObject *self, PyObject *args) {
    PyObject *d_obj, *e_obj;
    (void)self;
    if (!PyArg_ParseTuple(args, "OO", &d_obj, &e_obj)) return NULL;

    Py_buffer d_view, e_view;
    if (PyObject_GetBuffer(d_obj, &d_view, PyBUF_C_CONTIGUOUS | PyBUF_FORMAT | PyBUF_WRITABLE) < 0)
        return NULL;
    if (PyObject_GetBuffer(e_obj, &e_view, PyBUF_C_CONTIGUOUS | PyBUF_FORMAT) < 0) {
        PyBuffer_Release(&d_view);
        return NULL;
    }
    if (d_view.format == NULL || d_view.ndim != 1 || strcmp(d_view.format, "d") != 0) {
        PyErr_SetString(PyExc_ValueError, "d must be float64 (n,)");
        goto fail;
    }
    if (e_view.format == NULL || e_view.ndim != 1 || strcmp(e_view.format, "d") != 0 ||
        e_view.shape[0] != d_view.shape[0] - 1) {
        PyErr_SetString(PyExc_ValueError, "e must be float64 (n-1,)");
        goto fail;
    }

    const Py_ssize_t n = d_view.shape[0];
    double *d = (double *)d_view.buf;   /* eigenvalues overwrite d in-place */
    const double *e = (double *)e_view.buf;

    /* Work on a copy of e so the caller's buffer is untouched. */
    double *ew = PyMem_Malloc((size_t)(n - 1) * sizeof(double));
    double *Q = PyMem_Calloc((size_t)n * (size_t)n, sizeof(double));
    if (ew == NULL || Q == NULL) {
        PyErr_NoMemory();
        PyMem_Free(ew);
        PyMem_Free(Q);
        goto fail;
    }
    memcpy(ew, e, (size_t)(n - 1) * sizeof(double));
    for (Py_ssize_t i = 0; i < n; i++) Q[i * n + i] = 1.0;

    /* QR iteration with Wilkinson shift. */
    for (Py_ssize_t m = n - 1; m > 0; m--) {
        Py_ssize_t iter = 0;
        while (iter < 30 * (int)n) {
            Py_ssize_t l = m;
            while (l > 0 && fabs(ew[l - 1]) >= 1e-15 * (fabs(d[l - 1]) + fabs(d[l]))) {
                l--;
            }
            if (l == m) break;

            double g = (d[l + 1] - d[l]) / (2.0 * ew[l]);
            double r = sqrt(g * g + 1.0);
            double s = (g >= 0.0) ? d[m] - ew[l] / (g + r) : d[m] - ew[l] / (g - r);

            /* Givens sweep. */
            double c = 1.0, sn = 0.0, p = 0.0;
            for (Py_ssize_t i = l; i < m; i++) {
                double f = sn * ew[i];
                double b_val = c * ew[i];
                if (fabs(f) >= fabs(d[i])) {
                    c = d[i] / f;
                    r = sqrt(c * c + 1.0);
                    ew[i + 1] = f * r;
                    sn = 1.0 / r;
                    d[i] = c * sn * d[i];
                } else {
                    sn = f / d[i];
                    r = sqrt(sn * sn + 1.0);
                    ew[i + 1] = d[i] * r;
                    c = 1.0 / r;
                    d[i] = sn * c * d[i];
                }
                f = sn * d[i + 1] - c * p;
                d[i + 1] = c * d[i + 1] + sn * p;
                p = f;
                for (Py_ssize_t k = 0; k < n; k++) {
                    double qk = Q[k * n + i];
                    double qk1 = Q[k * n + i + 1];
                    Q[k * n + i] = c * qk + sn * qk1;
                    Q[k * n + i + 1] = -sn * qk + c * qk1;
                }
            }
            d[l] -= s;
            iter++;
        }
    }

    /* Sort eigenvalues ascending (insertion sort — n is small). */
    for (Py_ssize_t i = 1; i < n; i++) {
        double key = d[i];
        Py_ssize_t j = i;
        while (j > 0 && d[j - 1] > key) {
            d[j] = d[j - 1];
            j--;
        }
        d[j] = key;
    }

    PyObject *w = PyBytes_FromStringAndSize((const char *)d, (Py_ssize_t)n * (Py_ssize_t)sizeof(double));
    PyObject *q = PyBytes_FromStringAndSize((const char *)Q, (Py_ssize_t)n * (Py_ssize_t)n * (Py_ssize_t)sizeof(double));
    PyMem_Free(ew);
    PyMem_Free(Q);
    PyBuffer_Release(&d_view);
    PyBuffer_Release(&e_view);
    if (w == NULL || q == NULL) {
        Py_XDECREF(w);
        Py_XDECREF(q);
        return NULL;
    }
    return Py_BuildValue("NN", w, q);

fail:
    PyBuffer_Release(&d_view);
    PyBuffer_Release(&e_view);
    return NULL;
}

static PyMethodDef fast_linop_methods[] = {
    {"solve_triangular", solve_triangular, METH_VARARGS,
     "solve_triangular(L, b) -> x\n\nForward substitution for lower-triangular L."},
    {"eigh_tridiag", eigh_tridiag, METH_VARARGS,
     "eigh_tridiag(d, e) -> (w, Q)\n\nSymmetric tridiagonal eigenproblem via QR."},
    {"_has_openmp", has_openmp, METH_NOARGS, "Return True if built with OpenMP."},
    {"_has_neon", has_neon, METH_NOARGS, "Return True if built with ARM NEON."},
    {NULL, NULL, 0, NULL},
};

static struct PyModuleDef fast_linop_module = {
    PyModuleDef_HEAD_INIT,
    "cds2._fast_linop",
    "Compiled linear-algebra kernels for cds2.linalg.",
    -1,
    fast_linop_methods,
};

PyMODINIT_FUNC
PyInit__fast_linop(void) {
    return PyModule_Create(&fast_linop_module);
}

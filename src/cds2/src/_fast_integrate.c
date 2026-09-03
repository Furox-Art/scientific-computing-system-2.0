/* Numerical-integration hot loops for cds2.integrate.
 *
 * Kernels:
 *   rk4_step(f, y, t, h, n)  -> y_next   (classic RK4, f(y,t) written by caller)
 *   trapz_batch(x, y)        -> integral  (composite trapezoid, uniform x)
 *
 * The RK4 kernel takes a pre-evaluated slope buffer (caller computes k1..k4
 * via Python callback) so the C code stays allocation-light. The trapezoid
 * kernel is fully self-contained.
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

/* Composite trapezoid rule for uniformly spaced x.
 * x[n] and y[n] are float64 C-contiguous. Returns the scalar integral.
 */
static PyObject *
trapz_batch(PyObject *self, PyObject *args) {
    PyObject *x_obj, *y_obj;
    (void)self;
    if (!PyArg_ParseTuple(args, "OO", &x_obj, &y_obj)) return NULL;

    Py_buffer x_view, y_view;
    if (PyObject_GetBuffer(x_obj, &x_view, PyBUF_C_CONTIGUOUS | PyBUF_FORMAT) < 0)
        return NULL;
    if (PyObject_GetBuffer(y_obj, &y_view, PyBUF_C_CONTIGUOUS | PyBUF_FORMAT) < 0) {
        PyBuffer_Release(&x_view);
        return NULL;
    }
    if (x_view.format == NULL || x_view.ndim != 1 || strcmp(x_view.format, "d") != 0 ||
        y_view.format == NULL || y_view.ndim != 1 || strcmp(y_view.format, "d") != 0 ||
        x_view.shape[0] != y_view.shape[0] || x_view.shape[0] < 2) {
        PyErr_SetString(PyExc_ValueError, "x and y must be float64 (n,) with n >= 2");
        goto fail;
    }

    const Py_ssize_t n = x_view.shape[0];
    const double *x = (const double *)x_view.buf;
    const double *y = (const double *)y_view.buf;
    const double h = x[1] - x[0];

    /* Verify uniform spacing (relative tolerance 1e-9). */
    for (Py_ssize_t i = 2; i < n; i++) {
        if (fabs((x[i] - x[i - 1]) - h) > 1e-9 * fabs(h)) {
            PyErr_SetString(PyExc_ValueError, "x must be uniformly spaced");
            goto fail;
        }
    }

    double acc = 0.5 * (y[0] + y[n - 1]);
    for (Py_ssize_t i = 1; i < n - 1; i++) acc += y[i];

    PyBuffer_Release(&x_view);
    PyBuffer_Release(&y_view);
    return PyFloat_FromDouble(acc * h);

fail:
    PyBuffer_Release(&x_view);
    PyBuffer_Release(&y_view);
    return NULL;
}

/* RK4 step: y_next = y + (h/6)*(k1 + 2*k2 + 2*k3 + k4).
 * The caller supplies the four slope vectors (each float64 (n,)) so the C
 * kernel does no Python callbacks. Returns y_next as a new bytes buffer.
 */
static PyObject *
rk4_step(PyObject *self, PyObject *args) {
    PyObject *y_obj, *k1_obj, *k2_obj, *k3_obj, *k4_obj;
    double h;
    (void)self;
    if (!PyArg_ParseTuple(args, "OOOOOd", &y_obj, &k1_obj, &k2_obj, &k3_obj, &k4_obj, &h))
        return NULL;

    Py_buffer views[5];
    PyObject *objs[5] = {y_obj, k1_obj, k2_obj, k3_obj, k4_obj};
    for (int i = 0; i < 5; i++) {
        if (PyObject_GetBuffer(objs[i], &views[i], PyBUF_C_CONTIGUOUS | PyBUF_FORMAT) < 0) {
            for (int j = 0; j < i; j++) PyBuffer_Release(&views[j]);
            return NULL;
        }
        if (views[i].format == NULL || views[i].ndim != 1 || strcmp(views[i].format, "d") != 0) {
            PyErr_SetString(PyExc_ValueError, "all inputs must be float64 (n,)");
            for (int j = 0; j <= i; j++) PyBuffer_Release(&views[j]);
            return NULL;
        }
        if (views[i].shape[0] != views[0].shape[0]) {
            PyErr_SetString(PyExc_ValueError, "all inputs must have the same length");
            for (int j = 0; j <= i; j++) PyBuffer_Release(&views[j]);
            return NULL;
        }
    }

    const Py_ssize_t n = views[0].shape[0];
    const double *y = (const double *)views[0].buf;
    const double *k1 = (const double *)views[1].buf;
    const double *k2 = (const double *)views[2].buf;
    const double *k3 = (const double *)views[3].buf;
    const double *k4 = (const double *)views[4].buf;

    double *out = PyMem_Malloc((size_t)n * sizeof(double));
    if (out == NULL) {
        PyErr_NoMemory();
        for (int i = 0; i < 5; i++) PyBuffer_Release(&views[i]);
        return NULL;
    }

    const double h6 = h / 6.0;
    for (Py_ssize_t i = 0; i < n; i++) {
        out[i] = y[i] + h6 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
    }

    for (int i = 0; i < 5; i++) PyBuffer_Release(&views[i]);
    PyObject *result = PyBytes_FromStringAndSize((const char *)out, (Py_ssize_t)n * (Py_ssize_t)sizeof(double));
    PyMem_Free(out);
    return result;
}

static PyMethodDef fast_integrate_methods[] = {
    {"trapz_batch", trapz_batch, METH_VARARGS,
     "trapz_batch(x, y) -> integral\n\nComposite trapezoid rule for uniform x."},
    {"rk4_step", rk4_step, METH_VARARGS,
     "rk4_step(y, k1, k2, k3, k4, h) -> y_next\n\nClassic RK4 step from precomputed slopes."},
    {NULL, NULL, 0, NULL},
};

static struct PyModuleDef fast_integrate_module = {
    PyModuleDef_HEAD_INIT,
    "cds2._fast_integrate",
    "Compiled integration kernels for cds2.integrate.",
    -1,
    fast_integrate_methods,
};

PyMODINIT_FUNC
PyInit__fast_integrate(void) {
    return PyModule_Create(&fast_integrate_module);
}

/* Signal-processing hot loops for cds2.signals.
 *
 * Kernels:
 *   convolve_1d(x, kernel) -> y   (direct 1-D convolution, "same" mode)
 *   sos_filter(b, a, x, zi) -> (y, zf)   (second-order-sections IIR)
 *
 * Both use the plain buffer protocol; no NumPy headers required.
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

/* 1-D convolution in "same" mode (output length == x length).
 * x[n] and kernel[k] are float64 C-contiguous. The kernel is centered; for
 * even-length kernels the extra sample is placed on the right.
 */
static PyObject *
convolve_1d(PyObject *self, PyObject *args) {
    PyObject *x_obj, *k_obj;
    (void)self;
    if (!PyArg_ParseTuple(args, "OO", &x_obj, &k_obj)) return NULL;

    Py_buffer x_view, k_view;
    if (PyObject_GetBuffer(x_obj, &x_view, PyBUF_C_CONTIGUOUS | PyBUF_FORMAT) < 0)
        return NULL;
    if (PyObject_GetBuffer(k_obj, &k_view, PyBUF_C_CONTIGUOUS | PyBUF_FORMAT) < 0) {
        PyBuffer_Release(&x_view);
        return NULL;
    }
    if (x_view.format == NULL || x_view.ndim != 1 || strcmp(x_view.format, "d") != 0 ||
        k_view.format == NULL || k_view.ndim != 1 || strcmp(k_view.format, "d") != 0) {
        PyErr_SetString(PyExc_ValueError, "x and kernel must be float64 (n,)");
        goto fail;
    }

    const Py_ssize_t n = x_view.shape[0];
    const Py_ssize_t klen = k_view.shape[0];
    const double *x = (const double *)x_view.buf;
    const double *k = (const double *)k_view.buf;
    double *y = PyMem_Calloc((size_t)n, sizeof(double));
    if (y == NULL) {
        PyErr_NoMemory();
        goto fail;
    }

    const Py_ssize_t half = klen / 2;
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (Py_ssize_t i = 0; i < n; i++) {
        double acc = 0.0;
        for (Py_ssize_t j = 0; j < klen; j++) {
            Py_ssize_t idx = i + j - half;
            if (idx >= 0 && idx < n) acc += k[j] * x[idx];
        }
        y[i] = acc;
    }

    PyObject *out = PyBytes_FromStringAndSize((const char *)y, (Py_ssize_t)n * (Py_ssize_t)sizeof(double));
    PyMem_Free(y);
    PyBuffer_Release(&x_view);
    PyBuffer_Release(&k_view);
    return out;

fail:
    PyBuffer_Release(&x_view);
    PyBuffer_Release(&k_view);
    return NULL;
}

/* Direct-Form II transposed SOS IIR filter.
 * b[3*n_sections], a[3*n_sections], x[n], zi[2*n_sections].
 * Returns (y, zf) where y[n] is the filtered signal and zf[2*n_sections] is
 * the final state (can be fed back as zi for streaming).
 */
static PyObject *
sos_filter(PyObject *self, PyObject *args) {
    PyObject *b_obj, *a_obj, *x_obj, *zi_obj;
    (void)self;
    if (!PyArg_ParseTuple(args, "OOOO", &b_obj, &a_obj, &x_obj, &zi_obj)) return NULL;

    Py_buffer views[4];
    PyObject *objs[4] = {b_obj, a_obj, x_obj, zi_obj};
    for (int i = 0; i < 4; i++) {
        if (PyObject_GetBuffer(objs[i], &views[i], PyBUF_C_CONTIGUOUS | PyBUF_FORMAT | PyBUF_WRITABLE) < 0) {
            for (int j = 0; j < i; j++) PyBuffer_Release(&views[j]);
            return NULL;
        }
        if (views[i].format == NULL || views[i].ndim != 1 || strcmp(views[i].format, "d") != 0) {
            PyErr_SetString(PyExc_ValueError, "all inputs must be float64 1-D arrays");
            for (int j = 0; j <= i; j++) PyBuffer_Release(&views[j]);
            return NULL;
        }
    }

    const Py_ssize_t sections = views[0].shape[0] / 3;
    if (views[0].shape[0] % 3 != 0 || views[1].shape[0] != views[0].shape[0] ||
        views[3].shape[0] != 2 * sections) {
        PyErr_SetString(PyExc_ValueError, "b and a must be 3*n_sections, zi must be 2*n_sections");
        for (int i = 0; i < 4; i++) PyBuffer_Release(&views[i]);
        return NULL;
    }

    const Py_ssize_t n = views[2].shape[0];
    const double *b = (const double *)views[0].buf;
    const double *a = (const double *)views[1].buf;
    const double *x = (const double *)views[2].buf;
    double *zi = (double *)views[3].buf;

    double *y = PyMem_Calloc((size_t)n, sizeof(double));
    if (y == NULL) {
        PyErr_NoMemory();
        for (int i = 0; i < 4; i++) PyBuffer_Release(&views[i]);
        return NULL;
    }

    /* State buffer: 2 values per section. */
    double *state = PyMem_Malloc((size_t)(2 * sections) * sizeof(double));
    if (state == NULL) {
        PyErr_NoMemory();
        PyMem_Free(y);
        for (int i = 0; i < 4; i++) PyBuffer_Release(&views[i]);
        return NULL;
    }
    memcpy(state, zi, (size_t)(2 * sections) * sizeof(double));

    for (Py_ssize_t i = 0; i < n; i++) {
        double sample = x[i];
        for (Py_ssize_t s = 0; s < sections; s++) {
            const double *bs = b + s * 3;
            const double *as = a + s * 3;
            double *st = state + s * 2;
            double w = sample - as[1] * st[0] - as[2] * st[1];
            y[i] = bs[0] * w + bs[1] * st[0] + bs[2] * st[1];
            st[1] = st[0];
            st[0] = w;
            sample = y[i];
        }
    }

    /* Copy final state back into zi. */
    memcpy(zi, state, (size_t)(2 * sections) * sizeof(double));
    PyMem_Free(state);

    PyObject *out = PyBytes_FromStringAndSize((const char *)y, (Py_ssize_t)n * (Py_ssize_t)sizeof(double));
    PyMem_Free(y);
    for (int i = 0; i < 4; i++) PyBuffer_Release(&views[i]);
    return out;
}

static PyMethodDef fast_signal_methods[] = {
    {"convolve_1d", convolve_1d, METH_VARARGS,
     "convolve_1d(x, kernel) -> y\n\n1-D convolution in 'same' mode."},
    {"sos_filter", sos_filter, METH_VARARGS,
     "sos_filter(b, a, x, zi) -> y\n\nDirect-Form II transposed SOS IIR filter."},
    {NULL, NULL, 0, NULL},
};

static struct PyModuleDef fast_signal_module = {
    PyModuleDef_HEAD_INIT,
    "cds2._fast_signal",
    "Compiled signal-processing kernels for cds2.signals.",
    -1,
    fast_signal_methods,
};

PyMODINIT_FUNC
PyInit__fast_signal(void) {
    return PyModule_Create(&fast_signal_module);
}

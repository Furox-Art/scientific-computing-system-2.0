/* Safe Lloyd iteration kernel for cds2.ml.KMeans. */
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

static int checked_mul_size(size_t a, size_t b, size_t *out) {
    if (a != 0 && b > SIZE_MAX / a) return 0;
    *out = a * b;
    return 1;
}

static PyObject *lloyd(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *points_obj, *centers_obj;
    int max_iter;
    double tol;
    if (!PyArg_ParseTuple(args, "OOid", &points_obj, &centers_obj, &max_iter, &tol)) return NULL;

    Py_buffer points_view, centers_view;
    if (PyObject_GetBuffer(points_obj, &points_view, PyBUF_C_CONTIGUOUS | PyBUF_FORMAT) < 0) return NULL;
    if (PyObject_GetBuffer(centers_obj, &centers_view,
                           PyBUF_C_CONTIGUOUS | PyBUF_FORMAT | PyBUF_WRITABLE) < 0) {
        PyBuffer_Release(&points_view);
        return NULL;
    }
    if (points_view.format == NULL || points_view.ndim != 2 || strcmp(points_view.format, "d") != 0) {
        PyErr_SetString(PyExc_ValueError, "points must be a C-contiguous float64 (n, d) array");
        goto fail_views;
    }
    if (centers_view.format == NULL || centers_view.ndim != 2 || strcmp(centers_view.format, "d") != 0 ||
        centers_view.shape[1] != points_view.shape[1]) {
        PyErr_SetString(PyExc_ValueError, "centers must be a C-contiguous float64 (k, d) array matching the point dimension");
        goto fail_views;
    }

    const Py_ssize_t n = points_view.shape[0];
    const Py_ssize_t d = points_view.shape[1];
    const Py_ssize_t k = centers_view.shape[0];
    const double *points = (const double *)points_view.buf;
    double *centers = (double *)centers_view.buf;
    if (n <= 0 || k <= 0 || k > n || d <= 0) {
        PyErr_SetString(PyExc_ValueError, "need 1 <= k <= n and d >= 1");
        goto fail_views;
    }
    if (n > INT_MAX) {
        PyErr_SetString(PyExc_OverflowError, "too many samples for the compiled kernel");
        goto fail_views;
    }
    if (max_iter < 1) {
        PyErr_SetString(PyExc_ValueError, "max_iter must be at least 1");
        goto fail_views;
    }
    if (!isfinite(tol) || tol < 0.0) {
        PyErr_SetString(PyExc_ValueError, "tol must be a non-negative finite number");
        goto fail_views;
    }
    for (Py_ssize_t i = 0; i < n; i++) {
        for (Py_ssize_t t = 0; t < d; t++) {
            if (!isfinite(points[i * d + t])) {
                PyErr_SetString(PyExc_ValueError, "points must contain only finite values");
                goto fail_views;
            }
        }
    }
    for (Py_ssize_t j = 0; j < k; j++) {
        for (Py_ssize_t t = 0; t < d; t++) {
            if (!isfinite(centers[j * d + t])) {
                PyErr_SetString(PyExc_ValueError, "centers must contain only finite values");
                goto fail_views;
            }
        }
    }

    size_t kd;
    if (!checked_mul_size((size_t)k, (size_t)d, &kd)) {
        PyErr_NoMemory();
        goto fail_views;
    }
    long long *labels = PyMem_Malloc((size_t)n * sizeof(long long));
    double *dmin = PyMem_Malloc((size_t)n * sizeof(double));
    double *sums = PyMem_Malloc(kd * sizeof(double));
    double *counts = PyMem_Malloc((size_t)k * sizeof(double));
    double *next_centers = PyMem_Malloc(kd * sizeof(double));
    if (labels == NULL || dmin == NULL || sums == NULL || counts == NULL || next_centers == NULL) {
        PyErr_NoMemory();
        goto fail_alloc;
    }

    int iterations = 0;
    const int row_count = (int)n;
    Py_BEGIN_ALLOW_THREADS
    for (int iter = 0; iter < max_iter; iter++) {
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
        for (int i = 0; i < row_count; i++) {
            const double *row = points + (Py_ssize_t)i * d;
            double best_dist = DBL_MAX;
            Py_ssize_t best_index = 0;
            for (Py_ssize_t j = 0; j < k; j++) {
                const double *center_row = centers + j * d;
                double dist = 0.0;
                for (Py_ssize_t t = 0; t < d; t++) {
                    const double diff = row[t] - center_row[t];
                    dist += diff * diff;
                }
                if (dist < best_dist) {
                    best_dist = dist;
                    best_index = j;
                }
            }
            labels[i] = (long long)best_index;
            dmin[i] = best_dist;
        }

        memset(sums, 0, kd * sizeof(double));
        memset(counts, 0, (size_t)k * sizeof(double));
        for (Py_ssize_t i = 0; i < n; i++) {
            const Py_ssize_t label = (Py_ssize_t)labels[i];
            double *sum_row = sums + label * d;
            const double *row = points + i * d;
            for (Py_ssize_t t = 0; t < d; t++) sum_row[t] += row[t];
            counts[label] += 1.0;
        }

        for (Py_ssize_t j = 0; j < k; j++) {
            if (counts[j] > 0.0) continue;
            Py_ssize_t farthest = -1;
            double farthest_dist = -1.0;
            for (Py_ssize_t i = 0; i < n; i++) {
                const Py_ssize_t old = (Py_ssize_t)labels[i];
                if (counts[old] > 1.0 && dmin[i] > farthest_dist) {
                    farthest_dist = dmin[i];
                    farthest = i;
                }
            }
            if (farthest < 0) continue;
            const Py_ssize_t old = (Py_ssize_t)labels[farthest];
            for (Py_ssize_t t = 0; t < d; t++) {
                sums[old * d + t] -= points[farthest * d + t];
                sums[j * d + t] = points[farthest * d + t];
            }
            counts[old] -= 1.0;
            counts[j] = 1.0;
            labels[farthest] = (long long)j;
            dmin[farthest] = -1.0;
        }

        double shift = 0.0;
        for (Py_ssize_t j = 0; j < k; j++) {
            double *center_row = centers + j * d;
            double *next_row = next_centers + j * d;
            const double *sum_row = sums + j * d;
            for (Py_ssize_t t = 0; t < d; t++) {
                next_row[t] = sum_row[t] / counts[j];
                const double diff = fabs(next_row[t] - center_row[t]);
                if (diff > shift) shift = diff;
            }
        }
        memcpy(centers, next_centers, kd * sizeof(double));
        iterations = iter + 1;
        if (shift < tol) break;
    }

#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < row_count; i++) {
        const double *row = points + (Py_ssize_t)i * d;
        double best_dist = DBL_MAX;
        Py_ssize_t best_index = 0;
        for (Py_ssize_t j = 0; j < k; j++) {
            const double *center_row = centers + j * d;
            double dist = 0.0;
            for (Py_ssize_t t = 0; t < d; t++) {
                const double diff = row[t] - center_row[t];
                dist += diff * diff;
            }
            if (dist < best_dist) {
                best_dist = dist;
                best_index = j;
            }
        }
        labels[i] = (long long)best_index;
    }
    Py_END_ALLOW_THREADS

    PyObject *labels_bytes = PyBytes_FromStringAndSize((const char *)labels, n * (Py_ssize_t)sizeof(long long));
    PyObject *centers_bytes = PyBytes_FromStringAndSize((const char *)centers, k * d * (Py_ssize_t)sizeof(double));
    PyMem_Free(labels); PyMem_Free(dmin); PyMem_Free(sums); PyMem_Free(counts); PyMem_Free(next_centers);
    PyBuffer_Release(&points_view); PyBuffer_Release(&centers_view);
    if (labels_bytes == NULL || centers_bytes == NULL) {
        Py_XDECREF(labels_bytes); Py_XDECREF(centers_bytes); return NULL;
    }
    return Py_BuildValue("NNi", labels_bytes, centers_bytes, iterations);

fail_alloc:
    PyMem_Free(labels); PyMem_Free(dmin); PyMem_Free(sums); PyMem_Free(counts); PyMem_Free(next_centers);
fail_views:
    PyBuffer_Release(&points_view); PyBuffer_Release(&centers_view); return NULL;
}

static PyMethodDef methods[] = {
    {"lloyd", lloyd, METH_VARARGS, "lloyd(points, centers_init, max_iter, tol) -> (labels_i64, centers_f64, iterations)"},
    {NULL, NULL, 0, NULL},
};
static struct PyModuleDef module = {PyModuleDef_HEAD_INIT, "cds2._fast_kmeans", "Compiled Lloyd kernel.", -1, methods};
PyMODINIT_FUNC PyInit__fast_kmeans(void) { return PyModule_Create(&module); }

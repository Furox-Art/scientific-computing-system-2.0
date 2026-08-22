/* Fast Lloyd iteration kernel for cds2.ml.KMeans.
 *
 * Compiled as an optional accelerator (cds2._fast_kmeans). When absent,
 * cds2 falls back to the equivalent NumPy implementation in ml.py.
 *
 * Interfaces use the plain buffer protocol so the build needs neither the
 * NumPy headers nor any third-party library.
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <float.h>
#include <stdint.h>
#include <string.h>

static PyObject *
lloyd(PyObject *self, PyObject *args)
{
    PyObject *points_obj;
    PyObject *centers_obj;
    int max_iter;
    double tol;

    if (!PyArg_ParseTuple(args, "OOid", &points_obj, &centers_obj,
                          &max_iter, &tol)) {
        return NULL;
    }

    Py_buffer points_view;
    Py_buffer centers_view;

    /* FORMAT is required: without it the exporter may leave view.format
     * NULL, and the dtype checks below would read a NULL string. */
    if (PyObject_GetBuffer(points_obj, &points_view,
                           PyBUF_CONTIG_RO | PyBUF_FORMAT) < 0) {
        return NULL;
    }
    if (PyObject_GetBuffer(centers_obj, &centers_view,
                           PyBUF_C_CONTIGUOUS | PyBUF_FORMAT |
                               PyBUF_WRITABLE) < 0) {
        PyBuffer_Release(&points_view);
        return NULL;
    }

    if (points_view.format == NULL || points_view.ndim != 2 ||
        strcmp(points_view.format, "d") != 0) {
        PyErr_SetString(PyExc_ValueError,
                        "points must be a C-contiguous float64 (n, d) array");
        goto fail_views;
    }
    if (centers_view.format == NULL || centers_view.ndim != 2 ||
        strcmp(centers_view.format, "d") != 0 ||
        centers_view.shape[1] != points_view.shape[1]) {
        PyErr_SetString(PyExc_ValueError,
                        "centers must be a C-contiguous float64 (k, d) array "
                        "matching the point dimension");
        goto fail_views;
    }

    const Py_ssize_t n = points_view.shape[0];
    const Py_ssize_t d = points_view.shape[1];
    const Py_ssize_t k = centers_view.shape[0];
    const double *restrict points = (const double *)points_view.buf;
    double *restrict centers = (double *)centers_view.buf;

    if (n <= 0 || k <= 0 || k > n || d <= 0) {
        PyErr_SetString(PyExc_ValueError,
                        "need 1 <= k <= n and d >= 1");
        goto fail_views;
    }

    long long *restrict labels = PyMem_Malloc((size_t)n * sizeof(long long));
    double *restrict dmin = PyMem_Malloc((size_t)n * sizeof(double));
    double *restrict sums = PyMem_Malloc((size_t)k * (size_t)d * sizeof(double));
    double *restrict counts = PyMem_Malloc((size_t)k * sizeof(double));
    double *restrict next_centers =
        PyMem_Malloc((size_t)k * (size_t)d * sizeof(double));

    if (labels == NULL || dmin == NULL || sums == NULL || counts == NULL ||
        next_centers == NULL) {
        PyErr_NoMemory();
        goto fail_alloc;
    }

    int iterations = 0;

    /* Hot loops touch no Python objects: drop the GIL so embedded Python
     * threads keep running. The pragma below fans rows across cores when
     * the module was built with OpenMP (Linux wheels); MSVC builds keep
     * it disabled because its legacy OpenMP rejects this loop shape. */
    int row_count = (int)n;
    Py_BEGIN_ALLOW_THREADS

    for (int iter = 0; iter < max_iter; iter++) {
        /* Assignment step - dominant cost, embarrassingly parallel. */
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
        for (int i = 0; i < row_count; i++) {
            const Py_ssize_t row_index = (Py_ssize_t)i;
            const double *row = points + row_index * d;
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

        /* Update step. */
        memset(sums, 0, (size_t)k * (size_t)d * sizeof(double));
        memset(counts, 0, (size_t)k * sizeof(double));
        for (Py_ssize_t i = 0; i < n; i++) {
            double *sum_row = sums + labels[i] * d;
            const double *row = points + i * d;
            for (Py_ssize_t t = 0; t < d; t++) {
                sum_row[t] += row[t];
            }
            counts[labels[i]] += 1.0;
        }

        /* Relocate empty clusters onto the worst-fit point. */
        for (Py_ssize_t j = 0; j < k; j++) {
            if (counts[j] > 0.0) {
                continue;
            }
            Py_ssize_t farthest = -1;
            double farthest_dist = -1.0;
            for (Py_ssize_t i = 0; i < n; i++) {
                if (dmin[i] > farthest_dist) {
                    farthest_dist = dmin[i];
                    farthest = i;
                }
            }
            if (farthest < 0) {
                continue;
            }
            memcpy(sums + j * d, points + farthest * d, (size_t)d * sizeof(double));
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
                if (diff > shift) {
                    shift = diff;
                }
            }
        }
        memcpy(centers, next_centers, (size_t)k * (size_t)d * sizeof(double));
        iterations = iter + 1;
        if (shift < tol) {
            break;
        }
    }

    Py_END_ALLOW_THREADS

    PyObject *labels_bytes = PyBytes_FromStringAndSize(
        (const char *)labels, (Py_ssize_t)n * (Py_ssize_t)sizeof(long long));
    PyObject *centers_bytes = PyBytes_FromStringAndSize(
        (const char *)centers, k * d * (Py_ssize_t)sizeof(double));

    PyMem_Free(labels);
    PyMem_Free(dmin);
    PyMem_Free(sums);
    PyMem_Free(counts);
    PyMem_Free(next_centers);
    PyBuffer_Release(&points_view);
    PyBuffer_Release(&centers_view);

    if (labels_bytes == NULL || centers_bytes == NULL) {
        Py_XDECREF(labels_bytes);
        Py_XDECREF(centers_bytes);
        return NULL;
    }
    return Py_BuildValue("NNi", labels_bytes, centers_bytes, iterations);

fail_alloc:
    PyMem_Free(labels);
    PyMem_Free(dmin);
    PyMem_Free(sums);
    PyMem_Free(counts);
    PyMem_Free(next_centers);
fail_views:
    PyBuffer_Release(&points_view);
    PyBuffer_Release(&centers_view);
    return NULL;
}

static PyMethodDef fast_kmeans_methods[] = {
    {"lloyd", lloyd, METH_VARARGS,
     "lloyd(points, centers_init, max_iter, tol) -> (labels_i64, centers_f64, iterations)\n\n"
     "Run Lloyd iterations until convergence or iteration cap."},
    {NULL, NULL, 0, NULL},
};

static struct PyModuleDef fast_kmeans_module = {
    PyModuleDef_HEAD_INIT,
    "cds2._fast_kmeans",
    "Compiled Lloyd kernel for cds2.ml.KMeans.",
    -1,
    fast_kmeans_methods,
};

PyMODINIT_FUNC
PyInit__fast_kmeans(void)
{
    return PyModule_Create(&fast_kmeans_module);
}

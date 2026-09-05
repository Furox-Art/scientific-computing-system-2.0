/* Hardened PageRank power-iteration kernel for cds2.graph. */
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

static int require_buffer(PyObject *obj, Py_buffer *view, int want_double, const char *name) {
    if (PyObject_GetBuffer(obj, view, PyBUF_CONTIG_RO | PyBUF_FORMAT) < 0) return 0;
    int format_ok = want_double
        ? (view->format != NULL && strcmp(view->format, "d") == 0)
        : (view->format != NULL && (strcmp(view->format, "q") == 0 || strcmp(view->format, "l") == 0));
    if (!format_ok || view->ndim != 1) {
        PyErr_Format(PyExc_ValueError, "%s must be a contiguous 1-D %s array", name,
                     want_double ? "float64" : "int64");
        PyBuffer_Release(view);
        return 0;
    }
    return 1;
}

static PyObject *iterate(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *indptr_obj, *indices_obj, *data_obj, *dangling_obj;
    int n_nodes, max_iter;
    double damping, tol;
    if (!PyArg_ParseTuple(args, "OOOidOid", &indptr_obj, &indices_obj, &data_obj,
                          &n_nodes, &damping, &dangling_obj, &max_iter, &tol)) return NULL;

    Py_buffer indptr_view, indices_view, data_view, dangling_view;
    if (!require_buffer(indptr_obj, &indptr_view, 0, "indptr")) return NULL;
    if (!require_buffer(indices_obj, &indices_view, 0, "indices")) { PyBuffer_Release(&indptr_view); return NULL; }
    if (!require_buffer(data_obj, &data_view, 1, "data")) { PyBuffer_Release(&indptr_view); PyBuffer_Release(&indices_view); return NULL; }
    if (!require_buffer(dangling_obj, &dangling_view, 0, "dangling")) {
        PyBuffer_Release(&indptr_view); PyBuffer_Release(&indices_view); PyBuffer_Release(&data_view); return NULL;
    }

    if (n_nodes <= 0 || indptr_view.shape[0] != (Py_ssize_t)n_nodes + 1) {
        PyErr_SetString(PyExc_ValueError, "indptr must have exactly n + 1 entries"); goto fail_views;
    }
    if (!isfinite(damping) || !(0.0 < damping && damping < 1.0)) {
        PyErr_SetString(PyExc_ValueError, "damping must be strictly between 0 and 1"); goto fail_views;
    }
    if (max_iter < 1) { PyErr_SetString(PyExc_ValueError, "max_iter must be at least 1"); goto fail_views; }
    if (!isfinite(tol) || tol <= 0.0) { PyErr_SetString(PyExc_ValueError, "tol must be a positive finite number"); goto fail_views; }

    const int64_t *indptr = (const int64_t *)indptr_view.buf;
    const int64_t *indices = (const int64_t *)indices_view.buf;
    const double *weights = (const double *)data_view.buf;
    const int64_t *dangling = (const int64_t *)dangling_view.buf;
    const Py_ssize_t nnz = indices_view.shape[0];
    if (data_view.shape[0] != nnz) { PyErr_SetString(PyExc_ValueError, "indices and data must have the same length"); goto fail_views; }
    if (indptr[0] != 0 || indptr[n_nodes] != (int64_t)nnz) {
        PyErr_SetString(PyExc_ValueError, "indptr must start at 0 and end at nnz"); goto fail_views;
    }
    for (int j = 0; j < n_nodes; j++) {
        if (indptr[j] < 0 || indptr[j] > indptr[j + 1] || indptr[j + 1] > (int64_t)nnz) {
            PyErr_SetString(PyExc_ValueError, "indptr must be monotone and bounded by nnz"); goto fail_views;
        }
    }
    for (Py_ssize_t p = 0; p < nnz; p++) {
        if (indices[p] < 0 || indices[p] >= n_nodes) {
            PyErr_SetString(PyExc_ValueError, "indices contain a node outside 0..n-1"); goto fail_views;
        }
        if (!isfinite(weights[p]) || weights[p] < 0.0) {
            PyErr_SetString(PyExc_ValueError, "data must contain finite non-negative weights"); goto fail_views;
        }
    }
    const Py_ssize_t n_dangling = dangling_view.shape[0];
    for (Py_ssize_t p = 0; p < n_dangling; p++) {
        if (dangling[p] < 0 || dangling[p] >= n_nodes) {
            PyErr_SetString(PyExc_ValueError, "dangling contains a node outside 0..n-1"); goto fail_views;
        }
    }

    double *rank = PyMem_Malloc((size_t)n_nodes * sizeof(double));
    double *next = PyMem_Malloc((size_t)n_nodes * sizeof(double));
    if (rank == NULL || next == NULL) { PyErr_NoMemory(); goto fail_alloc; }
    const double initial = 1.0 / (double)n_nodes;
    for (int j = 0; j < n_nodes; j++) rank[j] = initial;
    const double teleport = (1.0 - damping) / (double)n_nodes;
    int iterations = 0;

    Py_BEGIN_ALLOW_THREADS
    for (int iter = 0; iter < max_iter; iter++) {
        double dangling_mass = 0.0;
#if defined(_OPENMP)
#pragma omp parallel for reduction(+ : dangling_mass)
#endif
        for (Py_ssize_t p = 0; p < n_dangling; p++) dangling_mass += rank[dangling[p]];
        const double uniform_add = damping * dangling_mass / (double)n_nodes + teleport;
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
        for (int j = 0; j < n_nodes; j++) {
            double acc = 0.0;
            for (int64_t p = indptr[j]; p < indptr[j + 1]; p++) acc += weights[p] * rank[indices[p]];
            next[j] = damping * acc + uniform_add;
        }
        double delta = 0.0;
        for (int j = 0; j < n_nodes; j++) {
            const double diff = fabs(next[j] - rank[j]);
            if (diff > delta) delta = diff;
        }
        double *tmp = rank; rank = next; next = tmp;
        iterations = iter + 1;
        if (delta < tol) break;
    }
    Py_END_ALLOW_THREADS

    PyObject *rank_bytes = PyBytes_FromStringAndSize((const char *)rank,
        (Py_ssize_t)n_nodes * (Py_ssize_t)sizeof(double));
    PyMem_Free(rank); PyMem_Free(next);
    PyBuffer_Release(&indptr_view); PyBuffer_Release(&indices_view);
    PyBuffer_Release(&data_view); PyBuffer_Release(&dangling_view);
    if (rank_bytes == NULL) return NULL;
    return Py_BuildValue("Ni", rank_bytes, iterations);

fail_alloc:
    PyMem_Free(rank); PyMem_Free(next);
fail_views:
    PyBuffer_Release(&indptr_view); PyBuffer_Release(&indices_view);
    PyBuffer_Release(&data_view); PyBuffer_Release(&dangling_view); return NULL;
}

static PyMethodDef methods[] = {
    {"iterate", iterate, METH_VARARGS, "iterate(indptr, indices, data, n, damping, dangling, max_iter, tol)"},
    {NULL, NULL, 0, NULL},
};
static struct PyModuleDef module = {PyModuleDef_HEAD_INIT, "cds2._fast_pagerank", "Compiled PageRank kernel.", -1, methods};
PyMODINIT_FUNC PyInit__fast_pagerank(void) { return PyModule_Create(&module); }

/* Fast PageRank power-iteration kernel for cds2.graph.
 *
 * Compiled as an optional accelerator (cds2._fast_pagerank). When absent,
 * cds2 falls back to the equivalent SciPy sparse implementation in graph.py.
 *
 * Consumes the CSR arrays of the TRANSPOSED normalized weight matrix: row j
 * holds the incoming links of node j, so one plain row sweep computes the
 * full follow step without any transposition work per iteration.
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdint.h>
#include <string.h>

static int
require_buffer(PyObject *obj, Py_buffer *view, const char *expected_format,
               const char *name)
{
    if (PyObject_GetBuffer(obj, view, PyBUF_CONTIG_RO | PyBUF_FORMAT) < 0) {
        return 0;
    }
    if (view->format == NULL || strcmp(view->format, expected_format) != 0) {
        PyErr_Format(PyExc_ValueError,
                     "%s must be a contiguous %s array", name,
                     strcmp(expected_format, "d") == 0 ? "float64"
                                                       : "int64");
        PyBuffer_Release(view);
        return 0;
    }
    return 1;
}

static PyObject *
iterate(PyObject *self, PyObject *args)
{
    PyObject *indptr_obj;
    PyObject *indices_obj;
    PyObject *data_obj;
    PyObject *dangling_obj;
    int n_nodes;
    double damping;
    int max_iter;
    double tol;

    if (!PyArg_ParseTuple(args, "OOOidOid", &indptr_obj, &indices_obj,
                          &data_obj, &n_nodes, &damping, &dangling_obj,
                          &max_iter, &tol)) {
        return NULL;
    }

    Py_buffer indptr_view;
    Py_buffer indices_view;
    Py_buffer data_view;
    Py_buffer dangling_view;

    if (!require_buffer(indptr_obj, &indptr_view, "q", "indptr")) {
        return NULL;
    }
    if (!require_buffer(indices_obj, &indices_view, "q", "indices")) {
        PyBuffer_Release(&indptr_view);
        return NULL;
    }
    if (!require_buffer(data_obj, &data_view, "d", "data")) {
        PyBuffer_Release(&indptr_view);
        PyBuffer_Release(&indices_view);
        return NULL;
    }
    if (!require_buffer(dangling_obj, &dangling_view, "q", "dangling")) {
        PyBuffer_Release(&indptr_view);
        PyBuffer_Release(&indices_view);
        PyBuffer_Release(&data_view);
        return NULL;
    }

    if (n_nodes <= 0 || indptr_view.shape[0] != (Py_ssize_t)n_nodes + 1) {
        PyErr_SetString(PyExc_ValueError,
                        "indptr must have exactly n + 1 entries");
        goto fail_views;
    }
    if (!(0.0 < damping && damping < 1.0)) {
        PyErr_SetString(PyExc_ValueError,
                        "damping must be strictly between 0 and 1");
        goto fail_views;
    }

    const Py_ssize_t *restrict indptr =
        (const Py_ssize_t *)indptr_view.buf;
    const long long *restrict indices =
        (const long long *)indices_view.buf;
    /* int64 buffers expose "q" (long long); shape[0]-derived sizes only. */
    const Py_ssize_t nnz = indices_view.shape[0];
    (void)nnz;
    const double *restrict weights = (const double *)data_view.buf;
    const Py_ssize_t n_dangling = dangling_view.shape[0];
    const long long *restrict dangling =
        (const long long *)dangling_view.buf;

    double *restrict rank = PyMem_Malloc((size_t)n_nodes * sizeof(double));
    double *restrict next = PyMem_Malloc((size_t)n_nodes * sizeof(double));
    if (rank == NULL || next == NULL) {
        PyErr_NoMemory();
        goto fail_alloc;
    }

    const double initial = 1.0 / (double)n_nodes;
    for (int j = 0; j < n_nodes; j++) {
        rank[j] = initial;
    }
    const double teleport = (1.0 - damping) / (double)n_nodes;

    int iterations = 0;
    for (int iter = 0; iter < max_iter; iter++) {
        double dangling_mass = 0.0;
        for (Py_ssize_t idx = 0; idx < n_dangling; idx++) {
            dangling_mass += rank[dangling[idx]];
        }
        const double uniform_add =
            damping * dangling_mass / (double)n_nodes + teleport;

        for (int j = 0; j < n_nodes; j++) {
            double acc = 0.0;
            for (Py_ssize_t p = indptr[j]; p < indptr[j + 1]; p++) {
                acc += weights[p] * rank[indices[p]];
            }
            next[j] = damping * acc + uniform_add;
        }

        double delta = 0.0;
        for (int j = 0; j < n_nodes; j++) {
            const double diff = fabs(next[j] - rank[j]);
            if (diff > delta) {
                delta = diff;
            }
        }
        double *swap_tmp = rank;
        rank = next;
        next = swap_tmp;
        iterations = iter + 1;
        if (delta < tol) {
            break;
        }
    }

    PyObject *rank_bytes = PyBytes_FromStringAndSize(
        (const char *)rank, (Py_ssize_t)n_nodes * (Py_ssize_t)sizeof(double));

    PyMem_Free(rank);
    PyMem_Free(next);
    PyBuffer_Release(&indptr_view);
    PyBuffer_Release(&indices_view);
    PyBuffer_Release(&data_view);
    PyBuffer_Release(&dangling_view);

    if (rank_bytes == NULL) {
        return NULL;
    }
    return Py_BuildValue("Ni", rank_bytes, iterations);

fail_alloc:
    PyMem_Free(rank);
    PyMem_Free(next);
fail_views:
    PyBuffer_Release(&indptr_view);
    PyBuffer_Release(&indices_view);
    PyBuffer_Release(&data_view);
    PyBuffer_Release(&dangling_view);
    return NULL;
}

static PyMethodDef fast_pagerank_methods[] = {
    {"iterate", iterate, METH_VARARGS,
     "iterate(indptr, indices, data, n, damping, dangling, max_iter, tol)"
     " -> (rank_f64, iterations)\n\nRun PageRank power iteration on the"
     " transposed normalized CSR arrays."},
    {NULL, NULL, 0, NULL},
};

static struct PyModuleDef fast_pagerank_module = {
    PyModuleDef_HEAD_INIT,
    "cds2._fast_pagerank",
    "Compiled PageRank kernel for cds2.graph.",
    -1,
    fast_pagerank_methods,
};

PyMODINIT_FUNC
PyInit__fast_pagerank(void)
{
    return PyModule_Create(&fast_pagerank_module);
}

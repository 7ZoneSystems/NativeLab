#include <Python.h>

static PyObject *nl_aibuilder_estimate_tokens(PyObject *self, PyObject *args) {
    PyObject *text;
    Py_ssize_t length;
    Py_ssize_t tokens;
    (void)self;

    if (!PyArg_ParseTuple(args, "U", &text)) {
        return NULL;
    }
    length = PyUnicode_GetLength(text);
    if (length < 0) {
        return NULL;
    }
    if (length == 0) {
        return PyLong_FromLong(0);
    }
    tokens = (length + 3) / 4;
    if (tokens < 1) {
        tokens = 1;
    }
    return PyLong_FromSsize_t(tokens);
}

static PyObject *nl_aibuilder_json_span(PyObject *self, PyObject *args) {
    PyObject *text;
    Py_ssize_t length;
    Py_ssize_t i;
    Py_ssize_t start = -1;
    int depth = 0;
    int in_string = 0;
    int escaped = 0;
    (void)self;

    if (!PyArg_ParseTuple(args, "U", &text)) {
        return NULL;
    }
    length = PyUnicode_GetLength(text);
    if (length < 0) {
        return NULL;
    }

    for (i = 0; i < length; i++) {
        Py_UCS4 ch = PyUnicode_ReadChar(text, i);
        if (ch == (Py_UCS4)-1 && PyErr_Occurred()) {
            return NULL;
        }

        if (start < 0) {
            if (ch == (Py_UCS4)'{') {
                start = i;
                depth = 1;
            }
            continue;
        }

        if (in_string) {
            if (escaped) {
                escaped = 0;
            } else if (ch == (Py_UCS4)'\\') {
                escaped = 1;
            } else if (ch == (Py_UCS4)'"') {
                in_string = 0;
            }
            continue;
        }

        if (ch == (Py_UCS4)'"') {
            in_string = 1;
        } else if (ch == (Py_UCS4)'{') {
            depth++;
        } else if (ch == (Py_UCS4)'}') {
            depth--;
            if (depth == 0) {
                return Py_BuildValue("(nn)", start, i + 1);
            }
        }
    }

    Py_RETURN_NONE;
}

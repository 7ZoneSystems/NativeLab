#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <string.h>

static PyObject *empty_unicode(void) {
    return PyUnicode_FromString("");
}

static PyObject *attr_as_unicode(PyObject *obj, const char *name) {
    PyObject *value = PyObject_GetAttrString(obj, name);
    PyObject *text;
    if (value == NULL) {
        PyErr_Clear();
        return empty_unicode();
    }
    text = PyObject_Str(value);
    Py_DECREF(value);
    return text;
}

static PyObject *mapping_get_default(PyObject *mapping, const char *key, const char *fallback) {
    PyObject *value = PyMapping_GetItemString(mapping, (char *)key);
    if (value == NULL) {
        PyErr_Clear();
        return PyUnicode_FromString(fallback);
    }
    return value;
}

static PyObject *content_as_unicode(PyObject *content) {
    if (PyList_Check(content) || PyTuple_Check(content)) {
        PyObject *seq = PySequence_Fast(content, "content must be a sequence");
        PyObject *parts;
        PyObject *sep;
        PyObject *joined;
        Py_ssize_t i, n;
        if (seq == NULL) {
            return NULL;
        }
        n = PySequence_Fast_GET_SIZE(seq);
        parts = PyList_New(n);
        if (parts == NULL) {
            Py_DECREF(seq);
            return NULL;
        }
        for (i = 0; i < n; i++) {
            PyObject *item = PySequence_Fast_GET_ITEM(seq, i);
            PyObject *text_value = NULL;
            PyObject *text;
            if (PyMapping_Check(item)) {
                text_value = PyMapping_GetItemString(item, "text");
                if (text_value == NULL) {
                    PyErr_Clear();
                }
            }
            text = PyObject_Str(text_value ? text_value : item);
            Py_XDECREF(text_value);
            if (text == NULL) {
                Py_DECREF(parts);
                Py_DECREF(seq);
                return NULL;
            }
            PyList_SET_ITEM(parts, i, text);
        }
        Py_DECREF(seq);
        sep = PyUnicode_FromString("\n");
        if (sep == NULL) {
            Py_DECREF(parts);
            return NULL;
        }
        joined = PyUnicode_Join(sep, parts);
        Py_DECREF(sep);
        Py_DECREF(parts);
        return joined;
    }
    return PyObject_Str(content);
}

static int append_piece(PyObject **target, PyObject *piece) {
    PyUnicode_Append(target, piece);
    return *target == NULL ? -1 : 0;
}

static int append_cstr(PyObject **target, const char *value) {
    PyObject *piece = PyUnicode_FromString(value);
    int result;
    if (piece == NULL) {
        return -1;
    }
    result = append_piece(target, piece);
    Py_DECREF(piece);
    return result;
}

static PyObject *nl_build_text_prompt(PyObject *self, PyObject *args) {
    PyObject *messages;
    PyObject *family;
    PyObject *seq;
    PyObject *out = NULL;
    PyObject *sys_buf = NULL;
    PyObject *bos = NULL;
    PyObject *user_prefix = NULL;
    PyObject *user_suffix = NULL;
    PyObject *assistant_prefix = NULL;
    PyObject *assistant_suffix = NULL;
    PyObject *result = NULL;
    Py_ssize_t i, n;

    (void)self;
    if (!PyArg_ParseTuple(args, "OO", &messages, &family)) {
        return NULL;
    }

    seq = PySequence_Fast(messages, "messages must be iterable");
    if (seq == NULL) {
        return NULL;
    }

    out = empty_unicode();
    sys_buf = empty_unicode();
    bos = attr_as_unicode(family, "bos");
    user_prefix = attr_as_unicode(family, "user_prefix");
    user_suffix = attr_as_unicode(family, "user_suffix");
    assistant_prefix = attr_as_unicode(family, "assistant_prefix");
    assistant_suffix = attr_as_unicode(family, "assistant_suffix");
    if (out == NULL || sys_buf == NULL || bos == NULL || user_prefix == NULL ||
        user_suffix == NULL || assistant_prefix == NULL || assistant_suffix == NULL) {
        goto done;
    }

    n = PySequence_Fast_GET_SIZE(seq);
    for (i = 0; i < n; i++) {
        PyObject *msg = PySequence_Fast_GET_ITEM(seq, i);
        PyObject *role_value;
        PyObject *content_value;
        PyObject *role_text = NULL;
        PyObject *content_text = NULL;
        int is_system = 0;
        int is_user = 0;
        int is_assistant = 0;

        if (!PyMapping_Check(msg)) {
            continue;
        }

        role_value = mapping_get_default(msg, "role", "user");
        content_value = mapping_get_default(msg, "content", "");
        if (role_value == NULL || content_value == NULL) {
            Py_XDECREF(role_value);
            Py_XDECREF(content_value);
            goto done;
        }

        role_text = PyObject_Str(role_value);
        content_text = content_as_unicode(content_value);
        Py_DECREF(role_value);
        Py_DECREF(content_value);
        if (role_text == NULL || content_text == NULL) {
            Py_XDECREF(role_text);
            Py_XDECREF(content_text);
            goto done;
        }

        is_system = PyUnicode_CompareWithASCIIString(role_text, "system") == 0;
        is_user = PyUnicode_CompareWithASCIIString(role_text, "user") == 0;
        is_assistant = PyUnicode_CompareWithASCIIString(role_text, "assistant") == 0;
        if (PyErr_Occurred()) {
            Py_DECREF(role_text);
            Py_DECREF(content_text);
            goto done;
        }

        if (is_system) {
            if (append_piece(&sys_buf, content_text) < 0 || append_cstr(&sys_buf, "\n") < 0) {
                Py_DECREF(role_text);
                Py_DECREF(content_text);
                goto done;
            }
        } else if (is_user) {
            Py_ssize_t sys_len = PyUnicode_GetLength(sys_buf);
            PyObject *user_text = NULL;
            PyObject *new_sys = NULL;
            if (sys_len < 0) {
                Py_DECREF(role_text);
                Py_DECREF(content_text);
                goto done;
            }
            if (sys_len > 0) {
                user_text = PyUnicode_Concat(sys_buf, content_text);
            } else {
                user_text = content_text;
                Py_INCREF(user_text);
            }
            if (user_text == NULL) {
                Py_DECREF(role_text);
                Py_DECREF(content_text);
                goto done;
            }
            if (append_piece(&out, user_prefix) < 0 ||
                append_piece(&out, user_text) < 0 ||
                append_piece(&out, user_suffix) < 0) {
                Py_DECREF(role_text);
                Py_DECREF(content_text);
                Py_DECREF(user_text);
                goto done;
            }
            Py_DECREF(user_text);
            new_sys = empty_unicode();
            if (new_sys == NULL) {
                Py_DECREF(role_text);
                Py_DECREF(content_text);
                goto done;
            }
            Py_SETREF(sys_buf, new_sys);
        } else if (is_assistant) {
            if (append_piece(&out, assistant_prefix) < 0 ||
                append_piece(&out, content_text) < 0 ||
                append_piece(&out, assistant_suffix) < 0) {
                Py_DECREF(role_text);
                Py_DECREF(content_text);
                goto done;
            }
        }

        Py_DECREF(role_text);
        Py_DECREF(content_text);
    }

    if (append_piece(&out, assistant_prefix) < 0) {
        goto done;
    }
    result = PyUnicode_Concat(bos, out);

done:
    Py_XDECREF(seq);
    Py_XDECREF(out);
    Py_XDECREF(sys_buf);
    Py_XDECREF(bos);
    Py_XDECREF(user_prefix);
    Py_XDECREF(user_suffix);
    Py_XDECREF(assistant_prefix);
    Py_XDECREF(assistant_suffix);
    return result;
}

static int object_to_long(PyObject *value, long fallback, long *out) {
    PyObject *num;
    if (value == NULL || value == Py_None) {
        *out = fallback;
        return 1;
    }
    num = PyNumber_Long(value);
    if (num == NULL) {
        PyErr_Clear();
        return 0;
    }
    *out = PyLong_AsLong(num);
    Py_DECREF(num);
    if (PyErr_Occurred()) {
        PyErr_Clear();
        return 0;
    }
    return 1;
}

static int object_to_double(PyObject *value, double fallback, double *out) {
    PyObject *num;
    if (value == NULL || value == Py_None) {
        *out = fallback;
        return 1;
    }
    num = PyNumber_Float(value);
    if (num == NULL) {
        PyErr_Clear();
        return 0;
    }
    *out = PyFloat_AsDouble(num);
    Py_DECREF(num);
    if (PyErr_Occurred()) {
        PyErr_Clear();
        return 0;
    }
    return 1;
}

static PyObject *sampler_payload_from_values(
    PyObject *top_k_obj,
    PyObject *min_p_obj,
    PyObject *typical_p_obj,
    PyObject *seed_obj
) {
    PyObject *dict = PyDict_New();
    PyObject *item;
    long top_k;
    long seed;
    double min_p;
    double typical_p;
    if (dict == NULL) {
        return NULL;
    }
    if (object_to_long(top_k_obj, 40, &top_k)) {
        if (top_k < 0) {
            top_k = 0;
        }
        item = PyLong_FromLong(top_k);
        if (item == NULL) {
            Py_DECREF(dict);
            return NULL;
        }
        if (PyDict_SetItemString(dict, "top_k", item) < 0) {
            Py_DECREF(item);
            Py_DECREF(dict);
            return NULL;
        }
        Py_DECREF(item);
    }
    if (object_to_double(min_p_obj, 0.0, &min_p) && min_p > 0.0) {
        item = PyFloat_FromDouble(min_p);
        if (item == NULL) {
            Py_DECREF(dict);
            return NULL;
        }
        if (PyDict_SetItemString(dict, "min_p", item) < 0) {
            Py_DECREF(item);
            Py_DECREF(dict);
            return NULL;
        }
        Py_DECREF(item);
    }
    if (object_to_double(typical_p_obj, 1.0, &typical_p) && typical_p > 0.0 && typical_p < 1.0) {
        item = PyFloat_FromDouble(typical_p);
        if (item == NULL) {
            Py_DECREF(dict);
            return NULL;
        }
        if (PyDict_SetItemString(dict, "typical_p", item) < 0) {
            Py_DECREF(item);
            Py_DECREF(dict);
            return NULL;
        }
        Py_DECREF(item);
    }
    if (object_to_long(seed_obj, -1, &seed) && seed >= 0) {
        item = PyLong_FromLong(seed);
        if (item == NULL) {
            Py_DECREF(dict);
            return NULL;
        }
        if (PyDict_SetItemString(dict, "seed", item) < 0) {
            Py_DECREF(item);
            Py_DECREF(dict);
            return NULL;
        }
        Py_DECREF(item);
    }
    return dict;
}

static PyObject *nl_sampler_payload(PyObject *self, PyObject *args, PyObject *kwargs) {
    static char *kwlist[] = {"top_k", "min_p", "typical_p", "seed", NULL};
    PyObject *top_k = NULL;
    PyObject *min_p = NULL;
    PyObject *typical_p = NULL;
    PyObject *seed = NULL;
    (void)self;
    if (!PyArg_ParseTupleAndKeywords(
            args, kwargs, "|OOOO", kwlist, &top_k, &min_p, &typical_p, &seed)) {
        return NULL;
    }
    return sampler_payload_from_values(top_k, min_p, typical_p, seed);
}

static int list_append_string(PyObject *list, const char *value) {
    PyObject *text = PyUnicode_FromString(value);
    int result;
    if (text == NULL) {
        return -1;
    }
    result = PyList_Append(list, text);
    Py_DECREF(text);
    return result;
}

static int list_append_object_as_string(PyObject *list, PyObject *value) {
    PyObject *text = PyObject_Str(value);
    int result;
    if (text == NULL) {
        return -1;
    }
    result = PyList_Append(list, text);
    Py_DECREF(text);
    return result;
}

static PyObject *nl_append_cli_sampler_args(PyObject *self, PyObject *args, PyObject *kwargs) {
    static char *kwlist[] = {"cmd", "top_k", "min_p", "typical_p", "seed", NULL};
    PyObject *cmd = NULL;
    PyObject *top_k = NULL;
    PyObject *min_p = NULL;
    PyObject *typical_p = NULL;
    PyObject *seed = NULL;
    PyObject *payload;
    PyObject *value;
    (void)self;
    if (!PyArg_ParseTupleAndKeywords(
            args, kwargs, "O|OOOO", kwlist, &cmd, &top_k, &min_p, &typical_p, &seed)) {
        return NULL;
    }
    if (!PyList_Check(cmd)) {
        PyErr_SetString(PyExc_TypeError, "cmd must be a list");
        return NULL;
    }
    payload = sampler_payload_from_values(top_k, min_p, typical_p, seed);
    if (payload == NULL) {
        return NULL;
    }
    value = PyDict_GetItemString(payload, "top_k");
    if (value != NULL &&
        (list_append_string(cmd, "--top-k") < 0 || list_append_object_as_string(cmd, value) < 0)) {
        Py_DECREF(payload);
        return NULL;
    }
    value = PyDict_GetItemString(payload, "min_p");
    if (value != NULL &&
        (list_append_string(cmd, "--min-p") < 0 || list_append_object_as_string(cmd, value) < 0)) {
        Py_DECREF(payload);
        return NULL;
    }
    value = PyDict_GetItemString(payload, "typical_p");
    if (value != NULL &&
        (list_append_string(cmd, "--typical") < 0 || list_append_object_as_string(cmd, value) < 0)) {
        Py_DECREF(payload);
        return NULL;
    }
    value = PyDict_GetItemString(payload, "seed");
    if (value != NULL &&
        (list_append_string(cmd, "--seed") < 0 || list_append_object_as_string(cmd, value) < 0)) {
        Py_DECREF(payload);
        return NULL;
    }
    Py_DECREF(payload);
    Py_RETURN_NONE;
}

static PyObject *nl_is_context_error(PyObject *self, PyObject *args) {
    PyObject *raw;
    PyObject *lower;
    const char *text;
    int found = 0;
    (void)self;
    if (!PyArg_ParseTuple(args, "U", &raw)) {
        return NULL;
    }
    lower = PyObject_CallMethod(raw, "lower", NULL);
    if (lower == NULL) {
        return NULL;
    }
    text = PyUnicode_AsUTF8(lower);
    if (text == NULL) {
        Py_DECREF(lower);
        return NULL;
    }
    found = strstr(text, "context size has been exceeded") != NULL ||
            strstr(text, "exceeds the available context size") != NULL ||
            strstr(text, "exceed_context_size") != NULL ||
            strstr(text, "context window") != NULL ||
            strstr(text, "n_ctx") != NULL;
    Py_DECREF(lower);
    if (found) {
        Py_RETURN_TRUE;
    }
    Py_RETURN_FALSE;
}

static PyObject *nl_build_reference_chunks(PyObject *self, PyObject *args) {
    PyObject *text;
    PyObject *chunks;
    Py_ssize_t step;
    Py_ssize_t overlap = 80;
    Py_ssize_t text_len;
    Py_ssize_t i = 0;
    (void)self;
    if (!PyArg_ParseTuple(args, "On|n", &text, &step, &overlap)) {
        return NULL;
    }
    if (!PyUnicode_Check(text)) {
        PyErr_SetString(PyExc_TypeError, "text must be str");
        return NULL;
    }
    if (step <= 0) {
        PyErr_SetString(PyExc_ValueError, "step must be greater than zero");
        return NULL;
    }
    if (overlap < 0) {
        overlap = 0;
    }
    text_len = PyUnicode_GetLength(text);
    if (text_len < 0) {
        return NULL;
    }
    chunks = PyList_New(0);
    if (chunks == NULL) {
        return NULL;
    }
    while (i < text_len) {
        Py_ssize_t end = i + step + overlap;
        PyObject *chunk;
        if (end > text_len) {
            end = text_len;
        }
        chunk = PyUnicode_Substring(text, i, end);
        if (chunk == NULL) {
            Py_DECREF(chunks);
            return NULL;
        }
        if (PyList_Append(chunks, chunk) < 0) {
            Py_DECREF(chunk);
            Py_DECREF(chunks);
            return NULL;
        }
        Py_DECREF(chunk);
        i += step;
    }
    return chunks;
}

#include "pipeline_core.c"
#include "../pipelinebuilder/aibuilder/aibuilder_core.c"

static PyMethodDef NativeCoreMethods[] = {
    {"build_text_prompt", nl_build_text_prompt, METH_VARARGS, "Build a text prompt from chat messages and a model family."},
    {"sampler_payload", (PyCFunction)nl_sampler_payload, METH_VARARGS | METH_KEYWORDS, "Normalize llama sampler options."},
    {"append_cli_sampler_args", (PyCFunction)nl_append_cli_sampler_args, METH_VARARGS | METH_KEYWORDS, "Append normalized sampler CLI flags to a list."},
    {"is_context_error", nl_is_context_error, METH_VARARGS, "Return True if text looks like a context-window error."},
    {"build_reference_chunks", nl_build_reference_chunks, METH_VARARGS, "Split reference text into overlapping chunks."},
    {"pipeline_normalize_ids", nl_pipeline_normalize_ids, METH_VARARGS, "Normalize pipeline block ids and remap connection endpoints."},
    {"pipeline_would_form_loop", nl_pipeline_would_form_loop, METH_VARARGS, "Return True if a new pipeline edge would create a loop."},
    {"pipeline_apply_transform", nl_pipeline_apply_transform, METH_VARARGS, "Apply a deterministic pipeline text transform."},
    {"pipeline_merge_texts", nl_pipeline_merge_texts, METH_VARARGS, "Merge pipeline branch outputs."},
    {"pipeline_route_edges", nl_pipeline_route_edges, METH_VARARGS, "Select routable pipeline edges and update visit counts."},
    {"pipeline_validate_records", nl_pipeline_validate_records, METH_VARARGS, "Validate pipeline records for execution readiness."},
    {"aibuilder_estimate_tokens", nl_aibuilder_estimate_tokens, METH_VARARGS, "Estimate AI pipeline builder tokens from text."},
    {"aibuilder_json_span", nl_aibuilder_json_span, METH_VARARGS, "Find the first balanced JSON object span in model text."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef nativecoremodule = {
    PyModuleDef_HEAD_INIT,
    "_native_core",
    "NativeLab C backend helper acceleration.",
    -1,
    NativeCoreMethods
};

PyMODINIT_FUNC PyInit__native_core(void) {
    return PyModule_Create(&nativecoremodule);
}

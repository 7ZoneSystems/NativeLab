/* Native pipeline hot-path helpers.
 *
 * This file is included by _core.c so local one-file extension builds keep
 * working, while pipeline-specific code stays isolated from backend helpers.
 */

static long pc_as_long(PyObject *value, long fallback, int *ok) {
    PyObject *num;
    long out;
    if (ok != NULL) {
        *ok = 0;
    }
    if (value == NULL || value == Py_None) {
        return fallback;
    }
    num = PyNumber_Long(value);
    if (num == NULL) {
        PyErr_Clear();
        return fallback;
    }
    out = PyLong_AsLong(num);
    Py_DECREF(num);
    if (PyErr_Occurred()) {
        PyErr_Clear();
        return fallback;
    }
    if (ok != NULL) {
        *ok = 1;
    }
    return out;
}

static int pc_set_contains_long(PyObject *set, long value) {
    PyObject *key = PyLong_FromLong(value);
    int contains;
    if (key == NULL) {
        return -1;
    }
    contains = PySet_Contains(set, key);
    Py_DECREF(key);
    return contains;
}

static int pc_set_add_long(PyObject *set, long value) {
    PyObject *key = PyLong_FromLong(value);
    int result;
    if (key == NULL) {
        return -1;
    }
    result = PySet_Add(set, key);
    Py_DECREF(key);
    return result;
}

static long pc_next_unused(long *counter, PyObject *reserved) {
    long next_id = *counter + 1;
    int contains;
    while ((contains = pc_set_contains_long(reserved, next_id)) > 0) {
        next_id++;
    }
    if (contains < 0) {
        PyErr_Clear();
    }
    *counter = next_id;
    return next_id;
}

static int pc_dict_set_if_missing(PyObject *dict, PyObject *key, long value) {
    PyObject *val;
    int contains;
    contains = PyDict_Contains(dict, key);
    if (contains < 0) {
        PyErr_Clear();
        return 0;
    }
    if (contains > 0) {
        return 0;
    }
    val = PyLong_FromLong(value);
    if (val == NULL) {
        return -1;
    }
    if (PyDict_SetItem(dict, key, val) < 0) {
        Py_DECREF(val);
        return -1;
    }
    Py_DECREF(val);
    return 0;
}

static PyObject *pc_map_endpoint(PyObject *remap, PyObject *endpoint) {
    PyObject *mapped = PyDict_GetItemWithError(remap, endpoint);
    if (mapped != NULL) {
        Py_INCREF(mapped);
        return mapped;
    }
    if (PyErr_Occurred()) {
        PyErr_Clear();
    }
    Py_INCREF(endpoint);
    return endpoint;
}

static PyObject *nl_pipeline_normalize_ids(PyObject *self, PyObject *args) {
    PyObject *ids_obj;
    PyObject *conns_obj;
    PyObject *ids_seq = NULL;
    PyObject *conns_seq = NULL;
    PyObject *reserved = NULL;
    PyObject *used = NULL;
    PyObject *remap = NULL;
    PyObject *out_ids = NULL;
    PyObject *out_conns = NULL;
    PyObject *result = NULL;
    Py_ssize_t n_ids;
    Py_ssize_t n_conns;
    Py_ssize_t i;
    long counter;
    (void)self;

    if (!PyArg_ParseTuple(args, "OOl", &ids_obj, &conns_obj, &counter)) {
        return NULL;
    }
    ids_seq = PySequence_Fast(ids_obj, "block ids must be iterable");
    conns_seq = PySequence_Fast(conns_obj, "connections must be iterable");
    if (ids_seq == NULL || conns_seq == NULL) {
        goto done;
    }

    n_ids = PySequence_Fast_GET_SIZE(ids_seq);
    n_conns = PySequence_Fast_GET_SIZE(conns_seq);
    reserved = PySet_New(NULL);
    used = PySet_New(NULL);
    remap = PyDict_New();
    out_ids = PyList_New(n_ids);
    out_conns = PyList_New(n_conns);
    if (reserved == NULL || used == NULL || remap == NULL ||
        out_ids == NULL || out_conns == NULL) {
        goto done;
    }

    for (i = 0; i < n_ids; i++) {
        PyObject *item = PySequence_Fast_GET_ITEM(ids_seq, i);
        int ok = 0;
        long bid = pc_as_long(item, 0, &ok);
        if (ok && bid > 0) {
            if (pc_set_add_long(reserved, bid) < 0) {
                goto done;
            }
            if (bid > counter) {
                counter = bid;
            }
        }
    }

    for (i = 0; i < n_ids; i++) {
        PyObject *item = PySequence_Fast_GET_ITEM(ids_seq, i);
        PyObject *py_bid;
        int ok = 0;
        int seen = 0;
        long original = pc_as_long(item, 0, &ok);
        long bid = original;
        if (ok && bid > 0) {
            seen = pc_set_contains_long(used, bid);
            if (seen < 0) {
                goto done;
            }
        }
        if (!ok || bid <= 0 || seen > 0) {
            bid = pc_next_unused(&counter, reserved);
            if (PyErr_Occurred()) {
                goto done;
            }
            if (pc_set_add_long(reserved, bid) < 0) {
                goto done;
            }
        }
        if (pc_set_add_long(used, bid) < 0) {
            goto done;
        }

        py_bid = PyLong_FromLong(bid);
        if (py_bid == NULL) {
            goto done;
        }
        PyList_SET_ITEM(out_ids, i, py_bid);

        if (pc_dict_set_if_missing(remap, item, bid) < 0) {
            goto done;
        }
        if (!ok || original != bid) {
            PyObject *new_key = PyLong_FromLong(bid);
            if (new_key == NULL) {
                goto done;
            }
            if (pc_dict_set_if_missing(remap, new_key, bid) < 0) {
                Py_DECREF(new_key);
                goto done;
            }
            Py_DECREF(new_key);
        }
    }

    for (i = 0; i < n_conns; i++) {
        PyObject *conn = PySequence_Fast_GET_ITEM(conns_seq, i);
        PyObject *pair_seq = PySequence_Fast(conn, "connection endpoints must be iterable");
        PyObject *from_obj;
        PyObject *to_obj;
        PyObject *mapped_from = NULL;
        PyObject *mapped_to = NULL;
        PyObject *pair = NULL;
        if (pair_seq == NULL) {
            goto done;
        }
        if (PySequence_Fast_GET_SIZE(pair_seq) < 2) {
            Py_DECREF(pair_seq);
            PyErr_SetString(PyExc_ValueError, "connection endpoint tuple must have two items");
            goto done;
        }
        from_obj = PySequence_Fast_GET_ITEM(pair_seq, 0);
        to_obj = PySequence_Fast_GET_ITEM(pair_seq, 1);
        mapped_from = pc_map_endpoint(remap, from_obj);
        mapped_to = pc_map_endpoint(remap, to_obj);
        Py_DECREF(pair_seq);
        if (mapped_from == NULL || mapped_to == NULL) {
            Py_XDECREF(mapped_from);
            Py_XDECREF(mapped_to);
            goto done;
        }
        pair = PyTuple_Pack(2, mapped_from, mapped_to);
        Py_DECREF(mapped_from);
        Py_DECREF(mapped_to);
        if (pair == NULL) {
            goto done;
        }
        PyList_SET_ITEM(out_conns, i, pair);
    }

    result = Py_BuildValue("{s:O,s:O,s:l}", "ids", out_ids, "connections", out_conns, "counter", counter);

done:
    Py_XDECREF(ids_seq);
    Py_XDECREF(conns_seq);
    Py_XDECREF(reserved);
    Py_XDECREF(used);
    Py_XDECREF(remap);
    Py_XDECREF(out_ids);
    Py_XDECREF(out_conns);
    return result;
}

static int pc_conn_pair(PyObject *conn, long *from_id, long *to_id) {
    PyObject *seq = PySequence_Fast(conn, "connection must be iterable");
    if (seq == NULL) {
        return -1;
    }
    if (PySequence_Fast_GET_SIZE(seq) < 2) {
        Py_DECREF(seq);
        PyErr_SetString(PyExc_ValueError, "connection tuple must have at least two items");
        return -1;
    }
    *from_id = pc_as_long(PySequence_Fast_GET_ITEM(seq, 0), 0, NULL);
    *to_id = pc_as_long(PySequence_Fast_GET_ITEM(seq, 1), 0, NULL);
    Py_DECREF(seq);
    return 0;
}

static PyObject *nl_pipeline_would_form_loop(PyObject *self, PyObject *args) {
    PyObject *conns_obj;
    PyObject *conns_seq = NULL;
    PyObject *visited = NULL;
    PyObject *queue = NULL;
    long from_bid;
    long to_bid;
    (void)self;

    if (!PyArg_ParseTuple(args, "Oll", &conns_obj, &from_bid, &to_bid)) {
        return NULL;
    }
    conns_seq = PySequence_Fast(conns_obj, "connections must be iterable");
    visited = PySet_New(NULL);
    queue = PyList_New(0);
    if (conns_seq == NULL || visited == NULL || queue == NULL) {
        goto error;
    }
    {
        PyObject *start = PyLong_FromLong(to_bid);
        if (start == NULL) {
            goto error;
        }
        if (PyList_Append(queue, start) < 0) {
            Py_DECREF(start);
            goto error;
        }
        Py_DECREF(start);
    }

    while (PyList_GET_SIZE(queue) > 0) {
        Py_ssize_t last = PyList_GET_SIZE(queue) - 1;
        PyObject *cur_obj = PyList_GET_ITEM(queue, last);
        long cur = pc_as_long(cur_obj, 0, NULL);
        PyObject *cur_key;
        Py_ssize_t i;
        if (PySequence_DelItem(queue, last) < 0) {
            goto error;
        }
        if (cur == from_bid) {
            Py_DECREF(conns_seq);
            Py_DECREF(visited);
            Py_DECREF(queue);
            Py_RETURN_TRUE;
        }
        cur_key = PyLong_FromLong(cur);
        if (cur_key == NULL) {
            goto error;
        }
        if (PySet_Contains(visited, cur_key) > 0) {
            Py_DECREF(cur_key);
            continue;
        }
        if (PySet_Add(visited, cur_key) < 0) {
            Py_DECREF(cur_key);
            goto error;
        }
        Py_DECREF(cur_key);
        for (i = 0; i < PySequence_Fast_GET_SIZE(conns_seq); i++) {
            PyObject *conn = PySequence_Fast_GET_ITEM(conns_seq, i);
            long c_from;
            long c_to;
            PyObject *to_obj;
            if (pc_conn_pair(conn, &c_from, &c_to) < 0) {
                goto error;
            }
            if (c_from != cur) {
                continue;
            }
            to_obj = PyLong_FromLong(c_to);
            if (to_obj == NULL) {
                goto error;
            }
            if (PyList_Append(queue, to_obj) < 0) {
                Py_DECREF(to_obj);
                goto error;
            }
            Py_DECREF(to_obj);
        }
    }
    Py_DECREF(conns_seq);
    Py_DECREF(visited);
    Py_DECREF(queue);
    Py_RETURN_FALSE;

error:
    Py_XDECREF(conns_seq);
    Py_XDECREF(visited);
    Py_XDECREF(queue);
    return NULL;
}

static PyObject *pc_mapping_get(PyObject *mapping, const char *key) {
    PyObject *value;
    if (mapping == NULL || !PyMapping_Check(mapping)) {
        Py_RETURN_NONE;
    }
    value = PyMapping_GetItemString(mapping, (char *)key);
    if (value == NULL) {
        PyErr_Clear();
        Py_RETURN_NONE;
    }
    return value;
}

static PyObject *pc_mapping_text(PyObject *mapping, const char *key, const char *fallback) {
    PyObject *value = pc_mapping_get(mapping, key);
    PyObject *text;
    if (value == NULL) {
        return NULL;
    }
    if (value == Py_None) {
        Py_DECREF(value);
        return PyUnicode_FromString(fallback);
    }
    text = PyObject_Str(value);
    Py_DECREF(value);
    return text;
}

static int pc_unicode_eq_cstr(PyObject *text, const char *value) {
    int cmp = PyUnicode_CompareWithASCIIString(text, value);
    if (cmp == -1 && PyErr_Occurred()) {
        PyErr_Clear();
        return 0;
    }
    return cmp == 0;
}

static PyObject *nl_pipeline_apply_transform(PyObject *self, PyObject *args) {
    PyObject *context_obj;
    PyObject *metadata;
    PyObject *context = NULL;
    PyObject *ttype = NULL;
    PyObject *result = NULL;
    (void)self;

    if (!PyArg_ParseTuple(args, "OO", &context_obj, &metadata)) {
        return NULL;
    }
    context = PyObject_Str(context_obj);
    ttype = pc_mapping_text(metadata, "transform_type", "prefix");
    if (context == NULL || ttype == NULL) {
        goto done;
    }

    if (pc_unicode_eq_cstr(ttype, "prefix")) {
        PyObject *val = pc_mapping_text(metadata, "transform_val", "");
        if (val != NULL) {
            result = PyUnicode_FromFormat("%U\n%U", val, context);
            Py_DECREF(val);
        }
    } else if (pc_unicode_eq_cstr(ttype, "suffix")) {
        PyObject *val = pc_mapping_text(metadata, "transform_val", "");
        if (val != NULL) {
            result = PyUnicode_FromFormat("%U\n%U", context, val);
            Py_DECREF(val);
        }
    } else if (pc_unicode_eq_cstr(ttype, "replace")) {
        PyObject *find = pc_mapping_text(metadata, "transform_find", "");
        PyObject *repl = pc_mapping_text(metadata, "transform_repl", "");
        if (find != NULL && repl != NULL) {
            if (PyUnicode_GetLength(find) > 0) {
                result = PyObject_CallMethod(context, "replace", "OO", find, repl);
            } else {
                result = context;
                Py_INCREF(result);
            }
        }
        Py_XDECREF(find);
        Py_XDECREF(repl);
    } else if (pc_unicode_eq_cstr(ttype, "upper")) {
        result = PyObject_CallMethod(context, "upper", NULL);
    } else if (pc_unicode_eq_cstr(ttype, "lower")) {
        result = PyObject_CallMethod(context, "lower", NULL);
    } else if (pc_unicode_eq_cstr(ttype, "strip")) {
        result = PyObject_CallMethod(context, "strip", NULL);
    } else if (pc_unicode_eq_cstr(ttype, "truncate")) {
        PyObject *val_obj = pc_mapping_get(metadata, "transform_val");
        int ok = 0;
        long n = 500;
        Py_ssize_t length;
        if (val_obj != NULL && val_obj != Py_None) {
            n = pc_as_long(val_obj, 500, &ok);
            if (!ok) {
                n = 500;
            }
        }
        Py_XDECREF(val_obj);
        if (n < 0) {
            n = 0;
        }
        length = PyUnicode_GetLength(context);
        if (length < 0) {
            goto done;
        }
        if (n > length) {
            n = (long)length;
        }
        result = PyUnicode_Substring(context, 0, (Py_ssize_t)n);
    } else {
        result = context;
        Py_INCREF(result);
    }

done:
    Py_XDECREF(context);
    Py_XDECREF(ttype);
    return result;
}

static PyObject *nl_pipeline_merge_texts(PyObject *self, PyObject *args) {
    PyObject *contexts_obj;
    PyObject *metadata;
    PyObject *contexts_seq = NULL;
    PyObject *parts = NULL;
    PyObject *mode = NULL;
    PyObject *sep = NULL;
    PyObject *result = NULL;
    Py_ssize_t i;
    Py_ssize_t n;
    int prepend = 0;
    (void)self;

    if (!PyArg_ParseTuple(args, "OO", &contexts_obj, &metadata)) {
        return NULL;
    }
    contexts_seq = PySequence_Fast(contexts_obj, "contexts must be iterable");
    mode = pc_mapping_text(metadata, "merge_mode", "concat");
    sep = pc_mapping_text(metadata, "merge_sep", "\n\n---\n\n");
    if (contexts_seq == NULL || mode == NULL || sep == NULL) {
        goto done;
    }
    n = PySequence_Fast_GET_SIZE(contexts_seq);
    parts = PyList_New(n);
    if (parts == NULL) {
        goto done;
    }
    prepend = pc_unicode_eq_cstr(mode, "prepend");
    for (i = 0; i < n; i++) {
        Py_ssize_t source_i = prepend ? (n - 1 - i) : i;
        PyObject *item = PySequence_Fast_GET_ITEM(contexts_seq, source_i);
        PyObject *text = PyObject_Str(item);
        if (text == NULL) {
            goto done;
        }
        PyList_SET_ITEM(parts, i, text);
    }
    if (pc_unicode_eq_cstr(mode, "json")) {
        PyObject *json_mod = PyImport_ImportModule("json");
        PyObject *dumps = NULL;
        PyObject *kwargs = NULL;
        PyObject *call_args = NULL;
        if (json_mod == NULL) {
            goto done;
        }
        dumps = PyObject_GetAttrString(json_mod, "dumps");
        Py_DECREF(json_mod);
        kwargs = Py_BuildValue("{s:i}", "indent", 2);
        call_args = PyTuple_Pack(1, parts);
        if (dumps == NULL || kwargs == NULL || call_args == NULL) {
            Py_XDECREF(dumps);
            Py_XDECREF(kwargs);
            Py_XDECREF(call_args);
            goto done;
        }
        result = PyObject_Call(dumps, call_args, kwargs);
        Py_DECREF(dumps);
        Py_DECREF(kwargs);
        Py_DECREF(call_args);
    } else {
        result = PyUnicode_Join(sep, parts);
    }

done:
    Py_XDECREF(contexts_seq);
    Py_XDECREF(parts);
    Py_XDECREF(mode);
    Py_XDECREF(sep);
    return result;
}

static int pc_string_ci_equal(const char *a, const char *b) {
    unsigned char ca;
    unsigned char cb;
    if (a == NULL || b == NULL) {
        return 0;
    }
    while (*a && *b) {
        ca = (unsigned char)*a;
        cb = (unsigned char)*b;
        if (ca >= 'A' && ca <= 'Z') {
            ca = (unsigned char)(ca + 32);
        }
        if (cb >= 'A' && cb <= 'Z') {
            cb = (unsigned char)(cb + 32);
        }
        if (ca != cb) {
            return 0;
        }
        a++;
        b++;
    }
    return *a == '\0' && *b == '\0';
}

static int pc_route_take(
    const char *mode,
    const char *branch_key,
    const char *port,
    PyObject *port_labels
) {
    if (mode == NULL || strcmp(mode, "all") == 0) {
        return 1;
    }
    if (strcmp(mode, "if") == 0) {
        if (strcmp(port, "E") != 0 && strcmp(port, "W") != 0) {
            return 1;
        }
        return (strcmp(port, "E") == 0 && strcmp(branch_key, "TRUE") == 0) ||
               (strcmp(port, "W") == 0 && strcmp(branch_key, "FALSE") == 0);
    }
    if (strcmp(mode, "switch") == 0 || strcmp(mode, "llm_switch") == 0) {
        PyObject *label_obj = NULL;
        PyObject *label_text = NULL;
        const char *label;
        int take;
        if (port_labels != NULL && PyMapping_Check(port_labels)) {
            label_obj = PyMapping_GetItemString(port_labels, (char *)port);
            if (label_obj == NULL) {
                PyErr_Clear();
            }
        }
        if (label_obj == NULL) {
            label_text = PyUnicode_FromString(port);
        } else {
            label_text = PyObject_Str(label_obj);
            Py_DECREF(label_obj);
        }
        if (label_text == NULL) {
            PyErr_Clear();
            return 0;
        }
        label = PyUnicode_AsUTF8(label_text);
        if (label == NULL) {
            Py_DECREF(label_text);
            PyErr_Clear();
            return 0;
        }
        if (strcmp(mode, "llm_switch") == 0) {
            take = pc_string_ci_equal(label, branch_key) || strcmp(label, "default") == 0 || label[0] == '\0';
        } else {
            take = strcmp(label, branch_key) == 0 || strcmp(label, "default") == 0 || label[0] == '\0';
        }
        Py_DECREF(label_text);
        return take;
    }
    if (strcmp(mode, "llm_score") == 0) {
        return strcmp(port, branch_key) == 0 ||
               strcmp(port, "N") == 0 ||
               (strcmp(port, "E") != 0 && strcmp(port, "S") != 0 &&
                strcmp(port, "W") != 0 && strcmp(port, "N") != 0);
    }
    return 1;
}

static PyObject *nl_pipeline_route_edges(PyObject *self, PyObject *args) {
    PyObject *records_obj;
    PyObject *visits;
    PyObject *mode_obj;
    PyObject *branch_key_obj;
    PyObject *port_labels;
    PyObject *records_seq = NULL;
    PyObject *out = NULL;
    long from_bid;
    const char *mode;
    const char *branch_key;
    Py_ssize_t i;
    (void)self;

    if (!PyArg_ParseTuple(args, "OlOOOO", &records_obj, &from_bid, &visits,
                          &mode_obj, &branch_key_obj, &port_labels)) {
        return NULL;
    }
    if (!PyDict_Check(visits)) {
        PyErr_SetString(PyExc_TypeError, "visit_counts must be a dict");
        return NULL;
    }
    mode = PyUnicode_AsUTF8(mode_obj);
    branch_key = PyUnicode_AsUTF8(branch_key_obj);
    if (mode == NULL || branch_key == NULL) {
        return NULL;
    }
    records_seq = PySequence_Fast(records_obj, "connection records must be iterable");
    out = PyList_New(0);
    if (records_seq == NULL || out == NULL) {
        goto done;
    }
    for (i = 0; i < PySequence_Fast_GET_SIZE(records_seq); i++) {
        PyObject *record = PySequence_Fast_GET_ITEM(records_seq, i);
        PyObject *rec_seq = PySequence_Fast(record, "connection record must be iterable");
        long idx;
        long rec_from;
        long rec_to;
        const char *from_port;
        int is_loop;
        long loop_times;
        char visit_key[96];
        PyObject *visit_obj;
        long visits_seen;
        long limit;
        PyObject *new_visits;
        PyObject *route_tuple;
        if (rec_seq == NULL) {
            goto done;
        }
        if (PySequence_Fast_GET_SIZE(rec_seq) < 6) {
            Py_DECREF(rec_seq);
            PyErr_SetString(PyExc_ValueError, "connection record must have six items");
            goto done;
        }
        idx = pc_as_long(PySequence_Fast_GET_ITEM(rec_seq, 0), 0, NULL);
        rec_from = pc_as_long(PySequence_Fast_GET_ITEM(rec_seq, 1), 0, NULL);
        from_port = PyUnicode_AsUTF8(PySequence_Fast_GET_ITEM(rec_seq, 2));
        rec_to = pc_as_long(PySequence_Fast_GET_ITEM(rec_seq, 3), 0, NULL);
        is_loop = PyObject_IsTrue(PySequence_Fast_GET_ITEM(rec_seq, 4));
        loop_times = pc_as_long(PySequence_Fast_GET_ITEM(rec_seq, 5), 1, NULL);
        Py_DECREF(rec_seq);
        if (from_port == NULL || is_loop < 0) {
            goto done;
        }
        if (rec_from != from_bid) {
            continue;
        }
        if (!pc_route_take(mode, branch_key, from_port, port_labels)) {
            continue;
        }
        snprintf(visit_key, sizeof(visit_key), "%ld->%ld", rec_from, rec_to);
        visit_obj = PyDict_GetItemString(visits, visit_key);
        visits_seen = visit_obj ? pc_as_long(visit_obj, 0, NULL) : 0;
        limit = is_loop ? loop_times : 1;
        if (limit < 1) {
            limit = 1;
        }
        if (visits_seen >= limit) {
            continue;
        }
        new_visits = PyLong_FromLong(visits_seen + 1);
        if (new_visits == NULL) {
            goto done;
        }
        if (PyDict_SetItemString(visits, visit_key, new_visits) < 0) {
            Py_DECREF(new_visits);
            goto done;
        }
        Py_DECREF(new_visits);
        route_tuple = Py_BuildValue("(lls)", idx, rec_to, from_port);
        if (route_tuple == NULL) {
            goto done;
        }
        if (PyList_Append(out, route_tuple) < 0) {
            Py_DECREF(route_tuple);
            goto done;
        }
        Py_DECREF(route_tuple);
    }

done:
    Py_XDECREF(records_seq);
    if (PyErr_Occurred()) {
        Py_XDECREF(out);
        return NULL;
    }
    return out;
}

static PyObject *pc_record_text(PyObject *record, const char *key, const char *fallback) {
    return pc_mapping_text(record, key, fallback);
}

static int pc_record_bool(PyObject *record, const char *key) {
    PyObject *value = pc_mapping_get(record, key);
    int truth = 0;
    if (value == NULL) {
        return 0;
    }
    truth = PyObject_IsTrue(value);
    Py_DECREF(value);
    if (truth < 0) {
        PyErr_Clear();
        return 0;
    }
    return truth > 0;
}

static PyObject *pc_label_message(const char *prefix, PyObject *label, const char *suffix) {
    PyObject *prefix_text = NULL;
    PyObject *label_text = NULL;
    PyObject *suffix_text = NULL;
    PyObject *tmp = NULL;
    PyObject *out = NULL;
    prefix_text = PyUnicode_FromString(prefix);
    label_text = PyObject_Str(label);
    suffix_text = PyUnicode_FromString(suffix);
    if (prefix_text == NULL || label_text == NULL || suffix_text == NULL) {
        goto done;
    }
    tmp = PyUnicode_Concat(prefix_text, label_text);
    if (tmp == NULL) {
        goto done;
    }
    out = PyUnicode_Concat(tmp, suffix_text);

done:
    Py_XDECREF(prefix_text);
    Py_XDECREF(label_text);
    Py_XDECREF(suffix_text);
    Py_XDECREF(tmp);
    return out;
}

static PyObject *nl_pipeline_validate_records(PyObject *self, PyObject *args) {
    PyObject *records_obj;
    PyObject *records_seq = NULL;
    Py_ssize_t connection_count;
    Py_ssize_t n;
    Py_ssize_t i;
    int has_input = 0;
    int has_output = 0;
    (void)self;

    if (!PyArg_ParseTuple(args, "On", &records_obj, &connection_count)) {
        return NULL;
    }
    records_seq = PySequence_Fast(records_obj, "records must be iterable");
    if (records_seq == NULL) {
        return NULL;
    }
    n = PySequence_Fast_GET_SIZE(records_seq);
    if (n <= 0) {
        Py_DECREF(records_seq);
        return PyUnicode_FromString("Canvas is empty - add blocks first.");
    }
    for (i = 0; i < n; i++) {
        PyObject *record = PySequence_Fast_GET_ITEM(records_seq, i);
        PyObject *btype = pc_record_text(record, "btype", "");
        if (btype == NULL) {
            Py_DECREF(records_seq);
            return NULL;
        }
        if (pc_unicode_eq_cstr(btype, "input")) {
            has_input = 1;
        } else if (pc_unicode_eq_cstr(btype, "output")) {
            has_output = 1;
        }
        Py_DECREF(btype);
    }
    if (!has_input) {
        Py_DECREF(records_seq);
        return PyUnicode_FromString("Pipeline needs at least one INPUT block.");
    }
    if (!has_output) {
        Py_DECREF(records_seq);
        return PyUnicode_FromString("Pipeline needs at least one OUTPUT block.");
    }
    if (connection_count <= 0) {
        Py_DECREF(records_seq);
        return PyUnicode_FromString("No connections drawn. Connect the blocks with arrows.");
    }

    for (i = 0; i < n; i++) {
        PyObject *record = PySequence_Fast_GET_ITEM(records_seq, i);
        PyObject *btype = pc_record_text(record, "btype", "");
        PyObject *label = pc_record_text(record, "label", "");
        PyObject *metadata = pc_mapping_get(record, "metadata");
        PyObject *message = NULL;
        if (btype == NULL || label == NULL || metadata == NULL) {
            Py_XDECREF(btype);
            Py_XDECREF(label);
            Py_XDECREF(metadata);
            Py_DECREF(records_seq);
            return NULL;
        }
        if (metadata == Py_None) {
            Py_DECREF(metadata);
            metadata = PyDict_New();
            if (metadata == NULL) {
                Py_DECREF(btype);
                Py_DECREF(label);
                Py_DECREF(records_seq);
                return NULL;
            }
        }

        if (pc_unicode_eq_cstr(btype, "reference") && !pc_record_bool(record, "has_ref_text")) {
            message = pc_label_message("Reference block '", label, "' has no text.\nRight-click it → Configure block…");
        } else if (pc_unicode_eq_cstr(btype, "knowledge") && !pc_record_bool(record, "has_knowledge_text")) {
            message = pc_label_message("Knowledge block '", label, "' has no text.\nRight-click it → Configure block…");
        } else if (pc_unicode_eq_cstr(btype, "pdf_summary") && !pc_record_bool(record, "has_pdf_path")) {
            message = pc_label_message("PDF block '", label, "' has no PDF selected.\nRight-click it → Configure block…");
        } else if (pc_unicode_eq_cstr(btype, "if_else") && !pc_record_bool(record, "has_condition")) {
            message = pc_label_message("IF/ELSE block '", label, "' has no condition set.\nRight-click it → Configure block…");
        } else if (pc_unicode_eq_cstr(btype, "switch") && !pc_record_bool(record, "has_switch_expr")) {
            message = pc_label_message("SWITCH block '", label, "' has no expression set.\nRight-click it → Configure block…");
        } else if (pc_unicode_eq_cstr(btype, "filter") && !pc_record_bool(record, "has_filter_cond")) {
            message = pc_label_message("FILTER block '", label, "' has no condition set.\nRight-click it → Configure block…");
        } else if (pc_unicode_eq_cstr(btype, "transform") && !pc_record_bool(record, "has_transform_type")) {
            message = pc_label_message("TRANSFORM block '", label, "' has no transform type set.\nRight-click it → Configure block…");
        } else if (pc_unicode_eq_cstr(btype, "custom_code") && !pc_record_bool(record, "has_custom_code")) {
            message = pc_label_message("Custom Code block '", label, "' has no code.\nRight-click it → Configure block…");
        } else if ((pc_unicode_eq_cstr(btype, "llm_if") ||
                    pc_unicode_eq_cstr(btype, "llm_switch") ||
                    pc_unicode_eq_cstr(btype, "llm_filter") ||
                    pc_unicode_eq_cstr(btype, "llm_transform") ||
                    pc_unicode_eq_cstr(btype, "llm_score")) &&
                   !pc_record_bool(record, "has_llm_instruction")) {
            message = pc_label_message("LLM logic block '", label, "' has no instruction.\nRight-click it → Configure block…");
        } else if ((pc_unicode_eq_cstr(btype, "llm_if") ||
                    pc_unicode_eq_cstr(btype, "llm_switch") ||
                    pc_unicode_eq_cstr(btype, "llm_filter") ||
                    pc_unicode_eq_cstr(btype, "llm_transform") ||
                    pc_unicode_eq_cstr(btype, "llm_score")) &&
                   !pc_record_bool(record, "llm_model_valid")) {
            message = pc_label_message("LLM logic block '", label, "' has no valid model attached.\nRight-click it → Configure block… and select a model.");
        } else if (pc_unicode_eq_cstr(btype, "model") && !pc_record_bool(record, "model_valid")) {
            message = pc_label_message("Model block '", label, "' has no valid model attached.\nDouble-click a model in the sidebar to add it.");
        }
        Py_DECREF(btype);
        Py_DECREF(label);
        Py_DECREF(metadata);
        if (message != NULL) {
            Py_DECREF(records_seq);
            return message;
        }
    }

    Py_DECREF(records_seq);
    Py_RETURN_NONE;
}

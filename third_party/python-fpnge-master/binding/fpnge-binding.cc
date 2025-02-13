#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "fpnge.h"

static PyObject* do_encode(const void* data, Py_ssize_t data_len, unsigned width, unsigned stride, unsigned height, unsigned num_channels, unsigned bits_per_channel, unsigned comp_level) {
	// check params
	if (width < 1 || height < 1) {
		PyErr_SetString(PyExc_ValueError, "Image size must be at least 1x1");
		return NULL;
	}
	if (num_channels < 1 || num_channels > 4) {
		PyErr_SetString(PyExc_ValueError, "Only 1-4 channels supported");
		return NULL;
	}
	if (bits_per_channel != 8 && bits_per_channel != 16) {
		PyErr_SetString(PyExc_ValueError, "Only 8/16 bits per channel supported");
		return NULL;
	}
	if (stride < width * num_channels * (bits_per_channel/8)) {
		PyErr_SetString(PyExc_ValueError, "Stride cannot be less than bytes per line");
		return NULL;
	}
	if (data_len < stride * height) {
		PyErr_SetString(PyExc_ValueError, "Not enough image data given");
		return NULL;
	}
	if (comp_level > FPNGE_COMPRESS_LEVEL_BEST) {
		#define _STR_HELPER(i) #i
		#define _STRINGIFY(i) _STR_HELPER(i)
		PyErr_SetString(PyExc_ValueError, "Compression level must be 0-" _STRINGIFY(FPNGE_COMPRESS_LEVEL_BEST));
		#undef _STR_HELPER
		#undef _STRINGIFY
		return NULL;
	}

	struct FPNGEOptions options;
	FPNGEFillOptions(&options, comp_level, FPNGE_CICP_NONE);
	
	size_t output_len = FPNGEOutputAllocSize(bits_per_channel/8, num_channels, width, height);
	PyObject *Py_output_buffer = PyBytes_FromStringAndSize(NULL, output_len + 1);
	if (!Py_output_buffer) {
		PyErr_NoMemory();
		return NULL;
	}
	PyBytesObject *sv = (PyBytesObject *)Py_output_buffer;
	
	Py_BEGIN_ALLOW_THREADS; // TODO: do we need to add ref onto the input buffer?
	output_len = FPNGEEncode(bits_per_channel/8, num_channels, data, width, stride, height, sv->ob_sval, &options);
	Py_END_ALLOW_THREADS;
	
	if (!output_len) {
		PyErr_SetString(PyExc_SystemError, "Failed to encode PNG");
		Py_DECREF(Py_output_buffer);
		return NULL;
	}
	
#if PY_MAJOR_VERSION == 3 && PY_MINOR_VERSION < 9
	Py_SIZE(sv) = output_len;
#else
	Py_SET_SIZE(sv, output_len);
#endif
	sv->ob_sval[output_len] = '\0';
	// Reset hash, this was removed in Python 3.11
#if PY_MAJOR_VERSION == 3 && PY_MINOR_VERSION < 11
	sv->ob_shash = -1;
#endif
	
	return Py_output_buffer;
}


PyObject* fpnge_encode_bytes(PyObject* self, PyObject* args) {
	(void)self;
	
	const char* data; Py_ssize_t data_len;
	unsigned width, stride = 0, height, num_channels, bits_per_channel, comp_level;
	
	if (!PyArg_ParseTuple(args, "y#IIIII|I", &data, &data_len, &width, &height, &num_channels, &bits_per_channel, &comp_level, &stride))
		return NULL;
	
	if (stride == 0) stride = width * num_channels * bits_per_channel/8;
	
	return do_encode(data, data_len, width, stride, height, num_channels, bits_per_channel, comp_level);
}

PyObject* fpnge_encode_view(PyObject* self, PyObject* args) {
	(void)self;
	
	PyObject* view;
	unsigned width, stride = 0, height, num_channels, bits_per_channel, comp_level;
	
	if (!PyArg_ParseTuple(args, "OIIIII|I", &view, &width, &height, &num_channels, &bits_per_channel, &comp_level, &stride))
		return NULL;

	Py_INCREF(view);

	if (!PyMemoryView_Check(view)) {
		PyErr_SetString(PyExc_SystemError, "Given object is not a memoryview");
		Py_DECREF(view);
		return NULL;
	}
	
	if (stride == 0) stride = width * num_channels * bits_per_channel/8;
	
	Py_buffer* view_buf = PyMemoryView_GET_BUFFER(view);
	if (!PyBuffer_IsContiguous(view_buf, 'C')) {
		PyErr_SetString(PyExc_SystemError, "Supplied memoryview must be C contiguous");
		Py_DECREF(view);
		return NULL;
	}
	
	PyObject* ret = do_encode(view_buf->buf, view_buf->len, width, stride, height, num_channels, bits_per_channel, comp_level);
	Py_DECREF(view);
	return ret;
}


static PyMethodDef fpnge_methods[] = {
	{
		"encode_bytes",
		fpnge_encode_bytes,
		METH_VARARGS,
		"encode_bytes(image_data, width, height, num_channels, bits_per_channel, comp_level, stride)"
	},
	{
		"encode_view",
		fpnge_encode_view,
		METH_VARARGS,
		"encode_view(image_data, width, height, num_channels, bits_per_channel, comp_level, stride)"
	},
	{NULL, NULL, 0, NULL}
};

static struct PyModuleDef fpnge_definition = {
	PyModuleDef_HEAD_INIT,
	"fpnge.binding",
	"Python binding for fpnge",
	-1,
	fpnge_methods
};

PyMODINIT_FUNC PyInit_binding(void) {
	Py_Initialize();
	PyObject* module = PyModule_Create(&fpnge_definition);
	PyModule_AddIntConstant(module, "FPNGE_COMPRESS_LEVEL_DEFAULT", FPNGE_COMPRESS_LEVEL_DEFAULT);
	return module;
}

import numpy as np
import ctypes
import os

lib = {}
lib_r = {}
lib_json = {}

dll_path = os.path.join('/', 'usr', 'lib', 'liboghma_py.so')
dll_lib = ctypes.CDLL(dll_path)


class my_data():
	def __init__(self, b, info, widget):
		self.units = b
		self.info = info
		self.defaults = None
		self.widget = widget
		self.units_widget = "QLabel"
		self.hidden = False
		self.hide_on_token_eq = None
		self.show_on_token_eq = None
		self.pack = []
		self.token = ""

class c_list(ctypes.Structure):
	_fields_ = [('names', ctypes.c_void_p),
				('len', ctypes.c_int),
				('len_max', ctypes.c_int)]

class hash_list(ctypes.Structure):
	_fields_ = [('names', ctypes.c_void_p),
				('data', ctypes.c_void_p),
				('index0', ctypes.c_void_p),
				('index1', ctypes.c_void_p),
				('len', ctypes.c_int),
				('len_max', ctypes.c_int),
				('item_size', ctypes.c_int),
				('free_function', ctypes.c_void_p),
				('thread_safe', ctypes.c_int),
				('all_data_valid', ctypes.c_int)]

fast_lib = hash_list()

class token_lib_item(ctypes.Structure):
	_fields_ = [('token', ctypes.c_char * 100),
				('english', ctypes.c_char * 100),
				('widget', ctypes.c_char * 100),
				('units', ctypes.c_char * 100),
				('log', ctypes.c_int),
				('hidden', ctypes.c_int),
				('hide_on_token', c_list),
				('hide_on_value', c_list),
				('show_on_token', c_list),
				('show_on_value', c_list),
				('default_token', c_list),
				('default_value', c_list),
				('pack', c_list)]

class tokens:

	def __init__(self):
		global lib
		global dll_lib

		self.lib = lib
		self.dll_lib = dll_lib

		self.dll_lib.token_lib_find.restype = ctypes.c_void_p
		if len(lib) == 0:
			self.build_token_lib()

	def build_token_lib(self):
		global fast_list
		#global lib
		self.dll_lib.token_lib_init(ctypes.byref(fast_lib))
		self.dll_lib.token_lib_build(ctypes.byref(fast_lib))

	def find(self, token):
		search_token = token.strip()
		ret = self.dll_lib.token_lib_find(ctypes.byref(fast_lib), ctypes.c_char_p(self.str2bytes(search_token)))
		if ret != None:
			tok = token_lib_item.from_address(ret)
			token = self.ctoken_to_python_token(tok)
			return token

		else:
			return False

	def ctoken_to_python_token(self, token):
		a = my_data(self.bytes2str(token.units), self.bytes2str(token.english), self.bytes2str(token.widget))
		a.token = self.bytes2str(token.token)
		a.hidden = bool(token.hidden)

		return token

	@staticmethod
	def bytes2str(string):
		ret=string
		try:
			ret=string.decode()
		except:
			pass
		return ret

	@staticmethod
	def isbytes(string):
		ret=string
		try:
			ret=string.decode()
			return True
		except:
			pass
		return False

	def str2bytes(self, string):
		if self.isbytes(string)==False:
			return string.encode()
		return string

class Label:

	def __init__(self, oghma_label):
		self.token = tokens()
		self.oghma_label = oghma_label

		if '.' in oghma_label:
			self.oghma_label = self.oghma_label.split('.')[-1]

		self.token = self.token.find(self.oghma_label)
		if self.token == False:
			self.english = 'Not Found'
			self.units = 'Not Found'
		else:
			self.english = self.token.english.decode()
			self.units = self.token.units.decode()
			self.widget = self.token.widget.decode()
			self.hidden = self.token.hidden
			self.pack = self.token.pack
			self.token = self.token.token.decode()

		latex_labels = ['^','_','{','}']
		r = []
		for l in latex_labels:
			if l in self.english:
				r.append(np.argwhere(l == np.asarray(list(self.english))).ravel()[0])
		#r = [for idx,l in enumerate(latex_labels)]
		r = np.asarray(r)

		if any(r):
			self.english = self.english[0:r[0]] + '$' + self.english[r[0]:r[-1]+1] + '$'





if __name__ == "__main__":
	print(Label("light_1.0.mue_jsc").english)
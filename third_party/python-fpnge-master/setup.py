#!/usr/bin/python3 -OO

import os
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext


class CustomBuild(build_ext):
	def build_extension(self, ext: Extension):
		if self.compiler.compiler_type == "msvc":
			ldflags = ["/OPT:REF", "/OPT:ICF"]
			cflags = ["/O2", "/GS-", "/Gy", "/sdl-", "/Oy", "/Oi"]
		else:
			ldflags = []
			cflags = [
				"-Wall",
				"-Wextra",
				"-fomit-frame-pointer",
				"-fno-rtti",
				"-fno-exceptions",
				"-O3",
				"-fPIC",
				"-fwrapv",
				"-std=c++14"
			]
		
		output_dir = os.path.dirname(self.build_lib)
		compiled_objects = []
		for source_files in [
			{
				"sources": ["binding/fpnge.cc"],
				"depends": ["binding/fpnge.h"],
				"gcc_flags": ["-march=native"],
			}
		]:
			args = {
				"sources": source_files["sources"],
				"output_dir": output_dir,
				"extra_postargs": cflags[:],
				"macros": [("NDEBUG", 1)]
			}
			if self.compiler.compiler_type == "msvc":
				if "msvc_flags" in source_files:
					args["extra_postargs"] += source_files["msvc_flags"]
			else:
				if "gcc_flags" in source_files:
					args["extra_postargs"] += source_files["gcc_flags"]

			if "include_dirs" in source_files:
				args["include_dirs"] = source_files["include_dirs"]
			if "macros" in source_files:
				args["macros"].extend(source_files["macros"])

			self.compiler.compile(**args)
			compiled_objects += self.compiler.object_filenames(source_files["sources"], output_dir=output_dir)

		# attach to Extension
		ext.extra_link_args = ldflags + compiled_objects
		ext.depends = compiled_objects

		# proceed with regular Extension build
		super(CustomBuild, self).build_extension(ext)

setup(
	name="fpnge",
	description="Python binding for fpnge",
	version="1.1.2",
	author="Anime Tosho",
	url="https://github.com/animetosho/python-fpnge/",
	license="CC0",
	ext_modules=[
		Extension("fpnge.binding", ["binding/fpnge-binding.cc"])
	],
	cmdclass={"build_ext": CustomBuild},
	packages=["fpnge"]
)

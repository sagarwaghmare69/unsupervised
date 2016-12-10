package = "unsupervised"
version = "scm-1"

source = {
   url = "git://github.com/sagarwaghmare69/unsupervised",
   tag = "master"
}

description = {
   summary = "Unsupervised learning methods",
   detailed = [[PCA, Factor Analysis, Expectation Maximization etc.]],
   homepage = "https://github.com/sagarwaghmare69/unsupervised",
   license = "BSD"
}

dependencies = {
   "torch >= 7.0",
}

build = {
   type = "command",
   build_command = [[
cmake -E make_directory build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$(LUAROCKS_PREFIX)" -DCMAKE_INSTALL_PREFIX="$(PREFIX)" && $(MAKE)
   ]],
   install_command = "cd build && $(MAKE) install"
}

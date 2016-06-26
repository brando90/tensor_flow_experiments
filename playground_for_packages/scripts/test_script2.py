import pkg_1.module1 as m1
# line bellow does not help to run f2. Notice that pip (package manager) points to
# my_proj directiory. This probably means it needs the package names to import things since sys.path points to
# my_project. So say the project name to help sys.path find your modules (that should be inside packages)
# import module1 as m1

m1.f2()

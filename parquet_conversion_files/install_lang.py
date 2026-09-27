import argostranslate.package as p
p.update_package_index()
pkg = next(x for x in p.get_available_packages() if x.from_code=='zh' and x.to_code=='en')
pkg.install()
print('zh->en installed')

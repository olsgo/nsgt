.PHONY: all test build_bdist build_sdist test_nsgt test_dist upload_pypi push_github clean

# all builds
all: build_bdist build_sdist

# all tests
test: test_nsgt test_dist
	
# build binary dist using modern build tools
# resultant *.whl file will be in subfolder dist
build_bdist:
	python3 -m build --wheel
	# for linux use auditwheel to convert to manylinux format
	if command -v auditwheel >/dev/null 2>&1 && auditwheel repair dist/nsgt*.whl; then \
		rm dist/nsgt*.whl; \
		mv wheelhouse/nsgt*.whl dist/; \
	fi
	
# build source dist using modern build tools
# resultant file will be in subfolder dist
build_sdist:
	python3 -m build --sdist

# test python module using pytest (modern approach)
test_nsgt:
	python3 -m pytest tests/ -v

# fallback to legacy test for compatibility
test_nsgt_legacy:
	python3 setup.py test

# test packages
test_dist:
	python3 -m twine check dist/*

# upload to pypi
upload_pypi:
	python3 -m twine upload --skip-existing dist/*

# push to github
push_github:
	-git remote add github https://$(GITHUB_ACCESS_TOKEN)@github.com/$(GITHUB_USERNAME)/nsgt.git
	# we need some extra treatment because the gitlab-runner doesn't check out the full history
	git push github HEAD:master --tags

# clean build artifacts
clean:
	rm -rf build/ dist/ *.egg-info/ __pycache__/ .pytest_cache/
	find . -name "*.pyc" -delete
	find . -name "*.pyo" -delete
	find . -name "*.so" -delete

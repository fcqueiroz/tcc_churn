.PHONY: clean train_model requirements create_environment test_environment

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = tcc_churn
$(eval PYTHON_INTERPRETER = $(shell which python3))

ifeq (,$(shell which conda))
HAS_CONDA=False
else
HAS_CONDA=True
endif

#################################################################################
# COMMANDS FOR MANAGING THE ENVIRONMENT                                         #
#                                                                               #
# !!! Recipes without any comments might not be ready for use                   #
# !!! Only the recipes visible through the 'help' command were checked          #
#                                                                               #
#################################################################################

create_environment:
ifeq (True,$(HAS_CONDA))
	@echo ">>> Detected conda, creating conda environment."
	conda create --name $(PROJECT_NAME) python=3
	@echo ">>> New conda env created. Activate with:\nsource activate $(PROJECT_NAME)"
else
	@pip install -q virtualenv virtualenvwrapper
	@echo ">>> Installing virtualenvwrapper if not already intalled.\nMake sure the following lines are in shell startup file\n\
	export WORKON_HOME=$$HOME/.virtualenvs\nexport PROJECT_HOME=$$HOME/Devel\nsource /usr/local/bin/virtualenvwrapper.sh\n"
	@bash -c "source `which virtualenvwrapper.sh`;mkvirtualenv $(PROJECT_NAME) --python=$(PYTHON_INTERPRETER)"
	@echo ">>> New virtualenv created. Activate with:\nworkon $(PROJECT_NAME)"
endif

requirements: test_environment
	pip install -U pip setuptools wheel
	pip install -r requirements.txt

test_environment:
	@echo $(AWS_name)
	@export AWS_name=redshifte; \
	$(PYTHON_INTERPRETER) test_environment.py

export_requirements:
	pip freeze > requirements.txt

## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

#################################################################################
# COMMANDS FOR MANAGING THE DATA PIPELINE                                       #
#################################################################################

data/raw/orders.csv: src/data/get_dataset.py
	@$(PYTHON_INTERPRETER) src/data/get_dataset.py orders

data/interim/orders.csv: src/data/make_dataset.py data/raw/orders.csv
	@$(PYTHON_INTERPRETER) src/data/make_dataset.py orders

data/processed/train.csv: src/features/build_features.py data/interim/orders.csv
	@$(PYTHON_INTERPRETER) src/features/build_features.py

data/processed/test.csv: src/features/build_features.py data/interim/orders.csv
	@$(PYTHON_INTERPRETER) src/features/build_features.py
	
models/0.1_decision_tree.sav: src/models/train_model.py data/processed/train.csv
	@echo "Training model..."
	@$(PYTHON_INTERPRETER) src/models/train_model.py

## Train Model
train_model:
	@make models/0.1_decision_tree.sav > /dev/null 
	@echo "Model is trained and up to date."

reports/performance.csv: src/models/predict_model.py data/processed/test.csv models/0.1_decision_tree.sav
	@echo "Testing model..."
	@$(PYTHON_INTERPRETER) src/models/predict_model.py

## Test Model
test_model: 
	@make reports/performance.csv > /dev/null 
	@echo "Performance score is up to date."


#################################################################################
# PROJECT RULES                                                                 #
#################################################################################



#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
.PHONY: help
help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')


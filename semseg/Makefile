SHELL := /bin/bash

MKFILE_PATH := $(abspath $(lastword $(MAKEFILE_LIST)))
MKFILE_DIR := $(dir $(MKFILE_PATH))

DOCKER_COMPOSE_FILES := \
	-f docker-compose.yml

DATA_DIR ?= $(MKFILE_DIR)/../data/
ASSETS_DIR ?= $(MKFILE_DIR)/../assets/
APPROACHES_DIR := $(MKFILE_DIR)/approaches/

PARAMETERS := MKFILE_DIR=$(MKFILE_DIR) \
	DATA_DIR=$(DATA_DIR) \
	ASSETS_DIR=$(ASSETS_DIR)

prepare-terminal-for-visualization:
	xhost +local:docker

####################
# BeyondBareQueries
####################
BBQ_ROOT_DIR := $(APPROACHES_DIR)/BeyondBareQueries/

build-bbq:
	cd $(MKFILE_DIR) && \
	$(PARAMETERS) \
	ROOT_DIR=$(BBQ_ROOT_DIR) \
	docker compose build bbq

run-bbq:
	cd $(MKFILE_DIR) && \
	$(PARAMETERS) \
	ROOT_DIR=$(BBQ_ROOT_DIR) \
	docker compose run bbq

####################
# ConceptGraphs
####################
CG_ROOT_DIR := $(APPROACHES_DIR)/ConceptGraphs/

build-concept-graphs:
	cd $(MKFILE_DIR) && \
	$(PARAMETERS) \
	ROOT_DIR=$(CG_ROOT_DIR) \
	docker compose build concept_graphs

run-concept-graphs:
	cd $(MKFILE_DIR) && \
	$(PARAMETERS) \
	ROOT_DIR=$(CG_ROOT_DIR) \
	docker compose run concept_graphs

####################
# OpenScene
####################
OPENSCENE_ROOT_DIR := $(APPROACHES_DIR)/OpenScene/

build-openscene:
	cd $(MKFILE_DIR) && \
	$(PARAMETERS) \
	ROOT_DIR=$(OPENSCENE_ROOT_DIR) \
	docker compose build openscene

run-openscene:
	cd $(MKFILE_DIR) && \
	$(PARAMETERS) \
	ROOT_DIR=$(OPENSCENE_ROOT_DIR) \
	docker compose run openscene
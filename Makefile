SHELL := /bin/bash

# -------------------------------------------------------
# Virtual environment
# -------------------------------------------------------
VENV_DIR ?= .venv
VENV_BIN := $(VENV_DIR)/bin
PY       := $(VENV_BIN)/python
PIP      := $(VENV_BIN)/pip


# -------------------------------------------------------
# Weights and checkpoints
# -------------------------------------------------------
WEIGHTS_DIR ?= weights
ViT_WEIGHT_FILE := moe_mae_pretrained_S_best.pth
ViT_WEIGHT_URL := https://huggingface.co/albughdadim/lightweight-metadata-aware-mixture-of-experts-mae-landsat/resolve/main/pretrained_S_best.pth?download=true

# -------------------------------------------------------
# Data directories
# -------------------------------------------------------
DATA_DIR ?= data
BIGEARTHNET_TEST_URL := https://object-store.os-api.cci2.ecmwf.int/meditwin-training/geo-moe-mae/bigearthnet_test.tar.gz
HF_EUROSAT_L_REPO ?= isaaccorley/eurosat-l
EUROSAT_L_DIR ?= $(DATA_DIR)/eurosat-l
EUROSAT_L_ARCHIVE ?= $(EUROSAT_L_DIR)/eurosat-l.tar.gz
EUROSAT_L_ZIP_SRC ?= $(DATA_DIR)/eurosat-l/eurosat-l.zip
# -------------------------------------------------------
# Virtualenv creation
# -------------------------------------------------------
$(VENV_DIR):
	python3 -m venv $(VENV_DIR)

# -------------------------------------------------------
# Install dependencies
# -------------------------------------------------------
install: $(VENV_DIR)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

# -------------------------------------------------------
# Download VIT weights
# -------------------------------------------------------
download-vit-weights:
	@mkdir -p $(WEIGHTS_DIR)
	curl -L "$(ViT_WEIGHT_URL)" -o $(WEIGHTS_DIR)/$(ViT_WEIGHT_FILE)
	@echo "ViT Weights downloaded to $(WEIGHTS_DIR)/$(ViT_WEIGHT_FILE)"

# -------------------------------------------------------
# Download EuroSAT-L data
# -------------------------------------------------------
download-eurosat-l: $(VENV_DIR)
	@mkdir -p $(DATA_DIR)
	@mkdir -p $(EUROSAT_L_DIR)
	@$(PY) -c "import huggingface_hub" >/dev/null 2>&1 || $(PIP) install huggingface_hub
	$(PY) -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='$(HF_EUROSAT_L_REPO)', repo_type='dataset', local_dir='$(EUROSAT_L_DIR)', local_dir_use_symlinks=False, resume_download=True)"
	@echo "EuroSAT-L downloaded to $(EUROSAT_L_DIR)"

# -------------------------------------------------------
# Extract EuroSAT-L data from local ZIP
# -------------------------------------------------------
extract-eurosat-l-zip:
	@test -f "$(EUROSAT_L_ZIP_SRC)" || (echo "Missing ZIP: $(EUROSAT_L_ZIP_SRC)" && exit 1)
	@mkdir -p $(EUROSAT_L_DIR)
	unzip -o "$(EUROSAT_L_ZIP_SRC)" -d "$(EUROSAT_L_DIR)"
	@echo "EuroSAT-L extracted to $(EUROSAT_L_DIR)"
	rm -f "$(EUROSAT_L_ZIP_SRC)"

# -------------------------------------------------------
# Download and extract EuroSAT-L data (if not already present)
# -------------------------------------------------------
download-extract-eurosat-l: download-eurosat-l extract-eurosat-l-zip

# -------------------------------------------------------
# Download BigEarthNet validation data
# -------------------------------------------------------
download-bigearthnet:
	@mkdir -p $(DATA_DIR)
	wget -O $(DATA_DIR)/bigearthnet_test.tar.gz "$(BIGEARTHNET_TEST_URL)"
	tar -xzf $(DATA_DIR)/bigearthnet_test.tar.gz -C $(DATA_DIR)
	rm -f $(DATA_DIR)/bigearthnet_test.tar.gz
	@echo "BigEarthNet test data downloaded and extracted to $(DATA_DIR)"

# -------------------------------------------------------
# Phony targets
# -------------------------------------------------------
.PHONY: install download-vit-weights download-eurosat-l extract-eurosat-l-zip download-extract-eurosat-l download-bigearthnet

# Makefile for building, signing, and loading satcp.ko

obj-m += satcp.o

KERNELDIR ?= /lib/modules/$(shell uname -r)/build
PWD       := $(shell pwd)

# Key file paths
KEYDIR    := /its/home/mm2350/bbr
KEY_PRIV  := $(KEYDIR)/mok.key
KEY_CERT  := $(KEYDIR)/mok.pem

MODULE := satcp.ko

all: modules sign load

# 1. Build the module
modules:
	$(MAKE) -C $(KERNELDIR) M=$(PWD) modules

# 2. Sign the module
sign:
	@echo "Signing module: $(MODULE)"
	@if [ -f "$(KEY_PRIV)" ] && [ -f "$(KEY_CERT)" ]; then \
	    $(KERNELDIR)/scripts/sign-file sha256 "$(KEY_PRIV)" "$(KEY_CERT)" "$(MODULE)"; \
	else \
	    echo "ERROR: Key files not found at $(KEY_PRIV) or $(KEY_CERT)"; \
	    exit 1; \
	fi

# 3. Load the module if not already loaded
load:
	@if ! lsmod | grep -q '^satcp '; then \
	    echo "Module 'satcp' not loaded; loading now..."; \
	    sudo insmod $(MODULE); \
	else \
	    echo "Module 'satcp' is already loaded."; \
	fi

clean:
	$(MAKE) -C $(KERNELDIR) M=$(PWD) clean

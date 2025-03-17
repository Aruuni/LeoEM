obj-m += satcp.o

KERNELDIR ?= /lib/modules/$(shell uname -r)/build
PWD       := $(shell pwd)

KEYDIR    := /path/to/your/key/if/you/have/one
KEY_PRIV  := $(KEYDIR)/mok.key
KEY_CERT  := $(KEYDIR)/mok.pem

MODULE := satcp.ko

# all: modules sign load
all: modules load

modules:
	$(MAKE) -C $(KERNELDIR) M=$(PWD) modules

sign:
	@echo "Signing module: $(MODULE)"
	@if [ -f "$(KEY_PRIV)" ] && [ -f "$(KEY_CERT)" ]; then \
	    $(KERNELDIR)/scripts/sign-file sha256 "$(KEY_PRIV)" "$(KEY_CERT)" "$(MODULE)"; \
	else \
	    echo "ERROR: Key files not found at $(KEY_PRIV) or $(KEY_CERT)"; \
	    exit 1; \
	fi

load:
	@if ! lsmod | grep -q '^satcp '; then \
	    echo "Module 'satcp' not loaded; loading now..."; \
	    sudo insmod $(MODULE); \
	else \
	    echo "Module 'satcp' is already loaded."; \
	fi

clean:
	$(MAKE) -C $(KERNELDIR) M=$(PWD) clean

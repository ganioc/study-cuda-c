

PROJECTS ?= $(shell find \
			$(shell ls -d */) \
			-name Makefile)

%.ph_build:
	+@$(MAKE) -C $(dir $*) $(MAKECMDGOALS)

%.ph_clean:
	+@$(MAKE) -C $(dir $*) clean $(USE_DEVICE)

all: $(addsuffix .ph_build,$(PROJECTS))
	@echo "Make all, building is Done."

.PHONY: test clean

test:
	echo "TEST"

	echo $(DIRECTORY)
	echo $(PROJECTS)

tidy:
	@find * | egrep "#" | xargs rm -f
	@find * | egrep "\~" | xargs rm -f
	@find * | egrep "nvvp" | xargs rm -f

clean: tidy $(addsuffix .ph_clean,$(PROJECTS))
	echo "CLEAN"
	

obj-m := dht20.o

all:
	$(MAKE) -C $(KERNEL_SRC) M=$(PWD) 

modules_install:
	$(MAKE) -C $(KERNEL_SRC) M=$(PWD) modules_install

clean:
	rm -f *.o *~ core .depend .*.cmd *.ko *.mod.c 
	rm -f Module.markers modules.order Module.symvers
	rm -rf .tmp_versions Modules.symvers

exec:	./xhpcg
	mkdir -p local && mkdir -p global
	rm -rf local/* && rm -rf global/*
	../../../../bin/mana_coordinator --daemon --config mana.conf
	mpirun -np 4 ../../../../bin/mana_launch --config mana.conf ./xhpcg

restart: ./xhpcg
	../../../../bin/mana_coordinator --daemon --config mana.conf
	mpirun -np 4 ../../../../bin/mana_restart --config mana.conf
clean: 
	rm -rf local/* && rm -rf global/*

#ifndef CONFIG_MPI_H
#define CONFIG_MPI_H

#include "../jalib/jassert.h"
#include "../jalib/jconvert.h"
#include <iostream>
#include <fstream>
#include <string>
#include <mpi.h>
#include <fcntl.h>
#include "util.h"
#include "util_config.h"
#include "constants.h"
#include <sys/types.h>
#include <sys/stat.h>


using namespace dmtcp;

struct UtilsMPI {

private:

  int _rank = -1;
  int _size = -1;

public:

  UtilsMPI();
  int getRank() const { return _rank; }
  int getSize() const { return _size; }
  char* getHostname(int test_mode);
  void getSystemTopology(int test_mode, Topology **topo);
  void performPartnerCopy(string ckptFilename, int *partnerMap);
  int checkCkptValid(int ckpt_type, string dir);
  string recoverFromCrash(ConfigInfo *cfg);
  static UtilsMPI instance();
};

#endif //ifndef CONFIG_MPI_H
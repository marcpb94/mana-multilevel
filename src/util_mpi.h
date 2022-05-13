#ifndef CONFIG_MPI_H
#define CONFIG_MPI_H

#include "../jalib/jassert.h"
#include "../jalib/jfilesystem.h"
#include "../jalib/jconvert.h"
#include "../jalib/jserialize.h"
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
#include <sys/mman.h>
#include <openssl/md5.h>
#include "dirent.h"
#include "uniquepid.h"
#include "mtcp/mtcp_header.h"
#include <sys/sysinfo.h>
#include <sched.h>

#define THREAD_FINISH 0
#define THREAD_NEXT 1

using namespace dmtcp;

struct ThreadInfo {
  int blockSize;
  int packetSize;
  int groupSize;
  int w;
  int **schedule;
  char **dataBlocks;
  char **encodedBlocks;
  int done;
  int finished;
  int next;
  pthread_mutex_t mutex[2];
  pthread_cond_t cond[2];
};


struct UtilsMPI {

private:

  int _rank = -1;
  int _size = -1;
  int _maxThreads = 0;

public:

  UtilsMPI();
  int getRank() const { return _rank; }
  int getSize() const { return _size; }
  char* getHostName(ConfigInfo *cfg);
  void getSystemTopology(ConfigInfo *cfg, Topology **topo);
  int getAvailableThreads(Topology *topo);
  void performPartnerCopy(string ckptFilename, Topology *topo);
  void performRSEncoding(string ckptFilename, Topology *topo);
  void performRSDecoding(string filename, string filenameEncoded, Topology* topo, int *to_recover, int *erasures, int total_success_raw, int *survivors);
  int jerasure_invert_matrix(int *mat, int *inv, int rows, int w);
  unsigned int countSetBits(unsigned char n);
  void setMatrixOnes(int* matrix, int w);
  void findGoodCauchyMatrix(int n, int w);
  void convertMatrixToBitmatrix(int *matrix, int *bitmatrix, int n, int w);
  int checkCkptValid(int ckpt_type, string dir, Topology *topo);
  int isCkptValid(const char *filename);
  int isEncodedCkptValid(const char *filename);
  int assistPartnerCopy(string ckptFilename, Topology *topo);
  int recoverFromPartnerCopy(string ckptFilename, Topology *topo);
  string recoverFromCrash(ConfigInfo *cfg);
  static UtilsMPI& instance();
};

#endif //ifndef CONFIG_MPI_H

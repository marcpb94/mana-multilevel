#include "util_mpi.h"

static UtilsMPI *_inst = NULL;

UtilsMPI::UtilsMPI(){
  int init;
  MPI_Initialized(&init);
  if(!init){
    MPI_Init(NULL, NULL);
  }
  MPI_Comm_rank(MPI_COMM_WORLD, &_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &_size);
}

UtilsMPI
UtilsMPI::instance(){
  if (_inst == NULL){
    _inst = new UtilsMPI();
  }
  return *_inst;
}

void
UtilsMPI::getSystemTopology(int test_mode, Topology **topo)
{
  char *hostname;
  int num_nodes;
  int mpi_size, mpi_rank;
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  hostname = UtilsMPI::instance().getHostname(test_mode);
  num_nodes = 0;
  //allocate enough memory for all hostnames
  char *allNodes = (char *)malloc(HOSTNAME_MAXSIZE * mpi_size);
  char *nameList = (char *)malloc(HOSTNAME_MAXSIZE * mpi_size);
  int *nodeMap = (int *)malloc(sizeof(int) * mpi_size);
  int *partnerMap = (int *)malloc(sizeof(int) * mpi_size);
  memset(nameList, 0, HOSTNAME_MAXSIZE * mpi_size);
  memset(nodeMap, 0, sizeof(int) * mpi_size);
  //distribute all hostnames
  MPI_Allgather(hostname, HOSTNAME_MAXSIZE, MPI_CHAR, allNodes,
                  HOSTNAME_MAXSIZE, MPI_CHAR, MPI_COMM_WORLD);

  //create rank-node mapping
  num_nodes = 0;
  int i, j, found;
  for (i = 0; i < mpi_size; i++){
    found = 0;
    //check if already in the list
    for (j = 0; j < num_nodes && !found; j++){
      if(strcmp(nameList + j*HOSTNAME_MAXSIZE, allNodes + i*HOSTNAME_MAXSIZE) == 0){
        found = 1;
        break;
      }
    }
    if (!found){
      //add node mapping
      nodeMap[i] = num_nodes;
      //add new node to nodelist
      strcpy(nameList + num_nodes*HOSTNAME_MAXSIZE, allNodes + i*HOSTNAME_MAXSIZE);
      num_nodes++;
    }
    else {
      //add node mapping
      nodeMap[i] = j;
    }
  }

  //initialize partner map with impossible value
  for(i = 0; i < mpi_size; i++) partnerMap[i] = -1;
  //create partner mapping
  for (i = 0; i < mpi_size; i++){
    if(partnerMap[i] == -1){
      found = 0;
      //find available partner rank in different node
      for(j = i+1; j < mpi_size; j++){
        if(nodeMap[i] != nodeMap[j] && partnerMap[j] == -1){
          //connect both nodes
          partnerMap[i] = j;
          partnerMap[j] = i;
          found = 1;
          break;
        }
      }
      JASSERT(found).Text("Could not complete partner mapping, application topology might contain uneven ranks per node or the execution is performed locally without enabling test mode.");
    }
  }

  *topo = new Topology(num_nodes, nameList, hostname, nodeMap, partnerMap);

  free(allNodes);
}

char *
UtilsMPI::getHostname(int test_mode)
{
  char *hostName = (char *)malloc(HOSTNAME_MAXSIZE);
  if(!test_mode){
    JASSERT(gethostname(hostName, sizeof hostName) == 0) (JASSERT_ERRNO);
  }
  else {
    //fake node size for testing purposes
    int node_size = 2;
    sprintf(hostName, "node%d", _rank/node_size);
  }
  return hostName;
}

string
UtilsMPI::recoverFromCrash(ConfigInfo *cfg){
  int rank = UtilsMPI::instance().getRank();
  printf("Test UtilsMPI rank: %d\n", rank);
  fflush(stdout);
  std::string restartDir = ConfigInfo::readRestartDir();
    if(!restartDir.empty()){
      printf("Found .restartdir with ckpt location %s\n", restartDir.c_str());
      fflush(stdout);
    }
    else {
      //else default to current directory
      restartDir = "./";
    }
    return string(restartDir.c_str());
}


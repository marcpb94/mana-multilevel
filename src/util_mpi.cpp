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

void
UtilsMPI::performPartnerCopy(string ckptFilename, int *partnerMap){
  printf("Starting to perform partner copy...\n");
  fflush(stdout);
  
  string partnerFilename = ckptFilename + "_partner";
  string partnerChksum = partnerFilename + "_md5chksum";
  string ckptChksum = ckptFilename + "_md5chksum";
  int myPartner = partnerMap[_rank];
  
  int fd_p = open(partnerFilename.c_str(), O_CREAT | O_TRUNC | O_WRONLY, S_IRUSR | S_IWUSR);
  int fd_m = open(ckptFilename.c_str(), O_RDONLY);
  JASSERT(fd_p != -1 && fd_m != -1) (fd_p)(fd_m);
  
  int fd_p_chksum = open(partnerChksum.c_str(), O_CREAT | O_TRUNC | O_WRONLY, S_IRUSR | S_IWUSR);
  int fd_m_chksum = open(ckptChksum.c_str(), O_RDONLY);
  JASSERT(fd_p_chksum != -1 && fd_m_chksum != -1) (fd_p)(fd_m);

  struct stat sb;
  JASSERT(fstat(fd_m, &sb) == 0);

  off_t ckptSize = sb.st_size, partnerCkptSize;
  off_t toSend = ckptSize, toRecv = 0;
  char *buff = (char *)malloc(DATA_BLOCK_SIZE);

  //decide send/recv order
  if(_rank > myPartner){
    //exchange ckpt file sizes
    MPI_Send(&ckptSize, sizeof(off_t), MPI_CHAR, myPartner, 0, MPI_COMM_WORLD);
    MPI_Recv(&partnerCkptSize, sizeof(off_t), MPI_CHAR, myPartner, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    toRecv = partnerCkptSize;
    //send ckpt file
    while (toSend > 0){
      off_t sendSize = (toSend > DATA_BLOCK_SIZE) ?
                           DATA_BLOCK_SIZE : toSend;

      Util::readAll(fd_m, buff, sendSize);
      MPI_Send(buff, sendSize, MPI_CHAR, myPartner, 0, MPI_COMM_WORLD);
      toSend -= sendSize;
    }
    //receive partner copy
    while(toRecv > 0){
      off_t recvSize = (toRecv > DATA_BLOCK_SIZE) ?
                           DATA_BLOCK_SIZE : toRecv;

      MPI_Recv(buff, recvSize, MPI_CHAR, myPartner, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      Util::writeAll(fd_p, buff, recvSize);
      toRecv -= recvSize;
    }
    //send checksum
    Util::readAll(fd_m_chksum, buff, 16);
    MPI_Send(buff, 16, MPI_CHAR, myPartner, 0, MPI_COMM_WORLD);
    //receive checksum
    MPI_Recv(buff, 16, MPI_CHAR, myPartner, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    Util::writeAll(fd_p_chksum, buff, 16);
  }
  else {
    //exchange ckpt file sizes
    MPI_Recv(&partnerCkptSize, sizeof(off_t), MPI_CHAR, myPartner, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    toRecv = partnerCkptSize;
    MPI_Send(&ckptSize, sizeof(off_t), MPI_CHAR, myPartner, 0, MPI_COMM_WORLD);
    //receive partner copy
    while(toRecv > 0){
      off_t recvSize = (toRecv > DATA_BLOCK_SIZE) ?
                           DATA_BLOCK_SIZE : toRecv;

      MPI_Recv(buff, recvSize, MPI_CHAR, myPartner, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      Util::writeAll(fd_p, buff, recvSize);
      toRecv -= recvSize;
    }
    //send ckpt file
    while (toSend > 0){
      off_t sendSize = (toSend > DATA_BLOCK_SIZE) ?
                           DATA_BLOCK_SIZE : toSend;

      Util::readAll(fd_m, buff, sendSize);
      MPI_Send(buff, sendSize, MPI_CHAR, myPartner, 0, MPI_COMM_WORLD);
      toSend -= sendSize;
    }
    //receive checksum
    MPI_Recv(buff, 16, MPI_CHAR, myPartner, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    Util::writeAll(fd_p_chksum, buff, 16);
    //send checksum
    Util::readAll(fd_m_chksum, buff, 16);
    MPI_Send(buff, 16, MPI_CHAR, myPartner, 0, MPI_COMM_WORLD);
  }

  printf("Finished performing partner copy.\n");
  fflush(stdout);

  free(buff);
  JASSERT(close(fd_p) == 0);
  JASSERT(close(fd_m) == 0);
  JASSERT(close(fd_p_chksum) == 0);
  JASSERT(close(fd_m_chksum) == 0);
}

char*
findCkptFilename(string ckpt_dir, string patternEnd){
  int dir_found = 0;  
  DIR *dir;
  struct dirent *entry;

  dir = opendir(ckpt_dir.c_str());
  if (dir == NULL) return 0;
  while ((entry = readdir(dir)) != NULL){
    
  }

  return NULL;
}

int
isCkptValid(char *filename){
  MD5_CTX context;
  unsigned char digest[16];
  string ckptFile = filename;
  string ckptChecksum = ckptFile + "_md5chksum";

  return 1;
}

int
UtilsMPI::checkCkptValid(int ckpt_type, string ckpt_dir){
  string real_dir = ckpt_dir + "/ckpt_rank_";
  real_dir += _rank + "/";
  int success = 0, allsuccess;  
  string ckptFilename;

  try {
    char *filename = findCkptFilename(real_dir, ".dmtcp");
    if(filename != NULL && isCkptValid(filename)){
        success = 1;
    }
    else {
      if(ckpt_type == CKPT_PARTNER){
        //TODO: also recover from partner if possible

      }
    }
  }
  catch (...){} //survive unexpected exceptions and assume failure

  //aggregate operation status from all ranks
  MPI_Allreduce(&success, &allsuccess, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

  // only return success if all ranks are successful
  return allsuccess;
}

string
UtilsMPI::recoverFromCrash(ConfigInfo *cfg){
    RestartInfo *restartInfo = new RestartInfo();
    restartInfo->readRestartInfo();
    string target = "";
    int i, found = 1, valid = 0, candidate;

    /*
    for(int i = 0; i <= CKPT_GLOBAL; i++){
      printf("type: %d, dir: %s, time: %ld\n", i, restartInfo->ckptDir[i].c_str(), restartInfo->ckptTime[i]);
    }
    fflush(stdout);
    */

    //recovery prioritizing by time of checkpoint
    while(found && !valid){
      found = 0;
      candidate = 0;
      for(i = 0; i <= CKPT_GLOBAL; i++){
        if(restartInfo->ckptTime[i] > 0 && (!found || 
              restartInfo->ckptTime[i] > restartInfo->ckptTime[candidate])){
        
          //update candidate
          candidate = i;
          found = 1;
        }
      }
      if(found){
        //check if the checkpoint is actually valid
        target = restartInfo->ckptDir[candidate];
        valid = checkCkptValid(candidate, target);
        if(!valid){
          //if not valid, set time to zero to invalidate
          //and loop again
          restartInfo->ckptTime[candidate] = 0;
        }
      }
    }

    JASSERT(found).Text("Restart point not found.");

    printf("Selected checkpoint type %d with location %s\n", candidate, target.c_str());
    fflush(stdout);

    return target;
}


#include "util_mpi.h"
#include "galois.h"
#include "jerasure.h"
#include <sys/time.h>
#include <math.h>
#include <algorithm>


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

static uint64_t
getRealCurrTime()
{
  struct timeval time;
  uint64_t microsec = 0;
  JASSERT(gettimeofday(&time, NULL) == 0);
  microsec = time.tv_sec*1000000L + time.tv_usec;
  return microsec;
}

void
UtilsMPI::getSystemTopology(ConfigInfo *cfg, Topology **topo)
{
  char *hostname;
  int num_nodes;
  int mpi_size, mpi_rank;
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  hostname = getHostName(cfg);
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

  int node_size = mpi_size / num_nodes;

  if (node_size != cfg->nodeSize && _rank == 0){
    printf("Real node size(%d) does not match specified value(%d), is this intended?\n", node_size, cfg->nodeSize);
    fflush(stdout);
  }

  // We first want to create an inverse node-process mapping as that done in Topology. There, nodeMap[i] linked
  // the ith process to the ith node in nameList. Instead, we want that in inverseNodeMap the node is indicated 
  // by the position in the array (e.g. the first n processes correspond to the first node, the next n processes
  // correspond to the second one, etc; where n is the number of processes per node). Then, inverseNodeMap[i] 
  // links the ith process to the node corresponding to its position in the array. This will the grouping of nodes
  // easier.

  int num_proc = mpi_size;

  int* inverseNodeMap = (int *)malloc(sizeof(int) * num_proc);

  // We initally set the mapping incorrectly
  for(i=0; i<num_proc; i++){
    inverseNodeMap[i] = -1;
  }

  for(i=0; i<num_proc; i++){
    int pos = nodeMap[i]*node_size;
    for(j=0;j<node_size;j++){
      // We search for an unoccupied space in the processes corresponding to a given node
      if(inverseNodeMap[pos+j]==-1){
        // inverseNodeMap[pos+j] is not occupied so we put process i here
        inverseNodeMap[pos+j]=i;
        break;
      }
    }
  }

  // Now all processes are nicely organized in inverseNodeMap depending on the node to which they belong. 
  // We want now to group different processes correponding to different nodes with a size set by group_size
  // and create communicators around them.
  int group_size = cfg->groupSize;
  MPI_Group newGroup, origGroup;
  MPI_Comm group_comm;
  int group_rank;
  int right, left;
  int my_node = nodeMap[mpi_rank];
  int section_ID = my_node / group_size;
  int buf = section_ID*group_size*node_size;
  int group[group_size];

  // Make sure that group size is multiple of the node size
  JASSERT(num_nodes % group_size == 0)(num_nodes)(group_size).Text("The number of nodes must be multiple of the group size.");

  // We have to find where our process is located in inverseNodeMap. We already now it corresponds to node nodeMap[mpi_rank]
  int pos;
  for(pos = 0;pos<node_size;pos++){
    if(inverseNodeMap[nodeMap[mpi_rank]*node_size+pos]==mpi_rank){
      break;
    }
  }

  for(i = 0;i<group_size;i++){
    group[i] = inverseNodeMap[buf+pos+i*node_size];
  }

  MPI_Comm_group(MPI_COMM_WORLD, &origGroup);
  MPI_Group_incl(origGroup, group_size, group, &newGroup);
  MPI_Comm_create(MPI_COMM_WORLD, newGroup, &group_comm); 
  MPI_Comm_rank(group_comm, &group_rank);
  right=(group_rank+1+group_size) % group_size;
  left=(group_rank-1+group_size) % group_size;


  MPI_Group_free(&origGroup);
  MPI_Group_free(&newGroup);

  //initialize partner map with impossible value
  for(i = 0; i < mpi_size; i++) partnerMap[i] = -1;
  //create partner mapping
  for (i = 0; i < mpi_size; i++){
    // The process is at node nodeMap[i]. Let us search at which position in inverseNodeMap.
    int buf = nodeMap[i]*node_size;
    int j;
    for(j=0;j<node_size;j++){
      if(i==inverseNodeMap[buf+j]){
        break;
      }
    }
    int sect = nodeMap[i] / group_size;
    partnerMap[i]= inverseNodeMap[sect*group_size*node_size + (buf+j+node_size)%(group_size*node_size)];
  }


  *topo = new Topology(num_nodes, nameList, hostname, nodeMap, partnerMap, mpi_size, node_size, group_size, section_ID, group_rank, right,
                      left, group_comm);

  free(allNodes);
}

char *
UtilsMPI::getHostName(ConfigInfo *cfg)
{
  char *hostName = (char *)malloc(HOSTNAME_MAXSIZE);
  if(!cfg->testMode){
    JASSERT(gethostname(hostName, HOSTNAME_MAXSIZE) == 0) (JASSERT_ERRNO);
  }
  else {
    //fake node size for testing purposes
    int node_size = cfg->nodeSize;
    sprintf(hostName, "node%d", _rank/node_size);
  }
  return hostName;
}

int
UtilsMPI::assistPartnerCopy(string ckptFilename, Topology *topo){
  
  printf("Rank %d | Group rank %d: performing recovery for partner group rank %d...\n", _rank, topo->groupRank, topo->left);
  fflush(stdout);
  
  string partnerCkptFile = ckptFilename;
  string partnerChksum = ckptFilename + "_md5chksum";
 
  int fd = open(partnerCkptFile.c_str(), O_RDONLY);
  if(fd == -1) return 0;
  
  int fd_chksum = open(partnerChksum.c_str(), O_RDONLY);
  if(fd_chksum == -1) return 0;

  struct stat sb;
  if(fstat(fd, &sb) != 0) return 0;
  
  off_t ckptSize = sb.st_size;
  off_t toSend = ckptSize;
  char *buff = (char *)malloc(DATA_BLOCK_SIZE);

  //exchange ckpt file sizes
  MPI_Send(&ckptSize, sizeof(off_t), MPI_CHAR, topo->left, 0, topo->groupComm);
  //send ckpt file
  while(toSend > 0){
    off_t sendSize = (toSend > DATA_BLOCK_SIZE) ?
			  DATA_BLOCK_SIZE : toSend;

    Util::readAll(fd, buff, sendSize);
    MPI_Send(buff, sendSize, MPI_CHAR, topo->left, 0, topo->groupComm);
    toSend -= sendSize;
  }
  //send checksum
  Util::readAll(fd_chksum, buff, 16);
  MPI_Send(buff, 16, MPI_CHAR, topo->left, 0, topo->groupComm);
 


  free(buff);
  close(fd);
  close(fd_chksum);

  return 1;
}

int
UtilsMPI::recoverFromPartnerCopy(string ckptFilename, Topology *topo){
  
  printf("Rank %d | Group rank %d: recovering from partner group rank %d...\n", _rank, topo->groupRank, topo->right);
  fflush(stdout);
  
  string ckptChksum = ckptFilename + "_md5chksum";
  
  int fd = open(ckptFilename.c_str(), O_CREAT | O_TRUNC | O_WRONLY, S_IRUSR | S_IWUSR);
  if(fd == -1) return 0;
  
  int fd_chksum = open(ckptChksum.c_str(), O_CREAT | O_TRUNC | O_WRONLY, S_IRUSR | S_IWUSR);
  if(fd_chksum == -1) return 0;

  off_t ckptSize;
  off_t toRecv;
  char *buff = (char *)malloc(DATA_BLOCK_SIZE);

  //exchange ckpt file sizes
  MPI_Recv(&ckptSize, sizeof(off_t), MPI_CHAR, topo->right, 0, topo->groupComm, MPI_STATUS_IGNORE);
  toRecv = ckptSize;
  //receive ckpt file
  while(toRecv > 0){
    off_t recvSize = (toRecv > DATA_BLOCK_SIZE) ?
			  DATA_BLOCK_SIZE : toRecv;

    MPI_Recv(buff, recvSize, MPI_CHAR, topo->right, 0, topo->groupComm, MPI_STATUS_IGNORE);
    Util::writeAll(fd, buff, recvSize);
    toRecv -= recvSize;
  }
  //receive checksum
  MPI_Recv(buff, 16, MPI_CHAR, topo->right, 0, topo->groupComm, MPI_STATUS_IGNORE);
  Util::writeAll(fd_chksum, buff, 16);


  free(buff);
  close(fd);
  close(fd_chksum);
 
  return 1;
}

void
UtilsMPI::performPartnerCopy(string ckptFilename, Topology *topo){

  if (_rank == 0){
    printf("Performing partner copy...\n");
    fflush(stdout);
  }
  
  string partnerFilename = ckptFilename + "_partner";
  string partnerChksum = partnerFilename + "_md5chksum";
  string ckptChksum = ckptFilename + "_md5chksum";
  
  int fd_p = open(partnerFilename.c_str(), O_CREAT | O_TRUNC | O_WRONLY, S_IRUSR | S_IWUSR);
  JASSERT(fd_p != -1);

  int fd_m = open(ckptFilename.c_str(), O_RDONLY);
  JASSERT(fd_m != -1);
  
  int fd_p_chksum = open(partnerChksum.c_str(), O_CREAT | O_TRUNC | O_WRONLY, S_IRUSR | S_IWUSR);
  JASSERT(fd_p_chksum != -1);

  int fd_m_chksum = open(ckptChksum.c_str(), O_RDONLY);
  JASSERT(fd_m_chksum != -1);

  struct stat sb;
  JASSERT(fstat(fd_m, &sb) == 0);

  off_t ckptSize = sb.st_size, partnerCkptSize;
  off_t toSend = ckptSize, toRecv = 0;
  char *buffRecv = (char *)malloc(DATA_BLOCK_SIZE);
  char *buffSend = (char *)malloc(DATA_BLOCK_SIZE);
  MPI_Request req;

  //identify left and right partners
  int group_rank, group_size;
  int right, left;
  right = topo->right;
  left = topo->left;

  //exchange ckpt file sizes
  MPI_Isend(&ckptSize, sizeof(off_t), MPI_CHAR, right, 0, topo->groupComm, &req);
  MPI_Recv(&partnerCkptSize, sizeof(off_t), MPI_CHAR, left, 0, topo->groupComm, MPI_STATUS_IGNORE);
  MPI_Wait(&req, MPI_STATUS_IGNORE);

  toRecv = partnerCkptSize;
  //send/recv ckpt file
  while (toSend > 0 || toRecv > 0){
    off_t sendSize = (toSend > DATA_BLOCK_SIZE) ?
                         DATA_BLOCK_SIZE : toSend;

    off_t recvSize = (toRecv > DATA_BLOCK_SIZE) ?
                         DATA_BLOCK_SIZE : toRecv;

    if (sendSize > 0) {
      Util::readAll(fd_m, buffSend, sendSize);
      MPI_Isend(buffSend, sendSize, MPI_CHAR, right, 0, topo->groupComm, &req);
    }
    if(recvSize > 0){
      MPI_Recv(buffRecv, recvSize, MPI_CHAR, left, 0, topo->groupComm, MPI_STATUS_IGNORE);
      Util::writeAll(fd_p, buffRecv, recvSize);
    }
    if(sendSize > 0){
      MPI_Wait(&req, MPI_STATUS_IGNORE);
    }
      
    toSend -= sendSize;
    toRecv -= recvSize;
  }

  Util::readAll(fd_m_chksum, buffSend, 16);
  MPI_Isend(buffSend, 16, MPI_CHAR, right, 0, topo->groupComm, &req);
  MPI_Recv(buffRecv, 16, MPI_CHAR, left, 0, topo->groupComm, MPI_STATUS_IGNORE);
  Util::writeAll(fd_p_chksum, buffRecv, 16);
  MPI_Wait(&req, MPI_STATUS_IGNORE);

  if (_rank == 0){
    printf("Finished partner copy.\n");
    fflush(stdout);
  }

  free(buffSend);
  free(buffRecv);
  JASSERT(close(fd_p) == 0);
  JASSERT(close(fd_m) == 0);
  JASSERT(close(fd_p_chksum) == 0);
  JASSERT(close(fd_m_chksum) == 0);
}

void 
UtilsMPI::performRSEncoding(string ckptFilename, Topology* topo){
  int w = 8;
  // Group rank 0 is the process that will perform RS encoding from the blocks of ckpt image the other processes send him.
  // After each block of encoded ckpt file has been calculated, group rank 0 sends to the other processes their corresponding part.
  if(topo->groupRank==0){
    printf("Performing RS encoding...\n");
    fflush(stdout);
  }

  string encodedFilename = ckptFilename + "_encoded";
  string encodedChksum = ckptFilename + "_encoded_md5chksum";
  string ckptChksum = ckptFilename + "_md5chksum";

  int fd_m = open(ckptFilename.c_str(), O_RDWR);
  JASSERT(fd_m != -1);

  int fd_e = open(encodedFilename.c_str(), O_CREAT | O_TRUNC | O_RDWR, S_IRUSR | S_IWUSR);
  JASSERT(fd_e != -1);

  struct stat sb;
  JASSERT(fstat(fd_m, &sb) == 0);
  off_t myCkptFileSize = sb.st_size;

  int i;
  off_t ckptFileSizes[topo->groupSize];
  if(topo->groupRank==0){
    ckptFileSizes[0]=myCkptFileSize;
    for(i=1;i<topo->groupSize;i++){
      MPI_Recv(&(ckptFileSizes[i]),sizeof(off_t),MPI_CHAR,i,0,topo->groupComm,MPI_STATUS_IGNORE);
    }
  }else{
    MPI_Send(&myCkptFileSize,sizeof(off_t),MPI_CHAR,0,0,topo->groupComm);
  }
  off_t maxSize;
  if(topo->groupRank==0){
    maxSize = ckptFileSizes[0];
    for(i=1;i<topo->groupSize;i++){
      if(ckptFileSizes[i]>maxSize){
        maxSize=ckptFileSizes[i];
      }
    }
    for(i=1;i<topo->groupSize;i++){
      MPI_Send(&maxSize,sizeof(off_t),MPI_CHAR,i,0,topo->groupComm);
    }
  }else{
    MPI_Recv(&maxSize,sizeof(off_t),MPI_CHAR,0,0,topo->groupComm,MPI_STATUS_IGNORE);
  }
  printf("Rank %d has ckptFileSize %lu\n",topo->groupRank,myCkptFileSize);
  fflush(stdout);

  // Now we have to elongate every ckpt file so that everyone has the same size. Their size is going to be maxSize+sizeof(off_t). 
  // Except that the size of each file has to be a multiple of w*sizeof(long), so we also have to round in this way

  off_t final_size = maxSize+sizeof(off_t);

  int packetSize = 16*1024*sizeof(long);
  int size = w*packetSize;

  if(final_size % (w*packetSize)!=0){
    final_size = (final_size / size)*size + size;
  }

  if(ftruncate(fd_m, final_size) == -1){
    JASSERT(0).Text("Error with truncation on ckpt image.\n");
  }
  
  // Now we will write the original size of the ckpt to the end of the elongated file, so that at restart we can recover 
  // the original ckpt image

  if(lseek(fd_m, -sizeof(off_t), SEEK_END) == -1){
    JASSERT(0).Text("Unable to seek in file.\n");
  }
  if(write(fd_m,&myCkptFileSize,sizeof(off_t)) == -1){
    JASSERT(0).Text("Unable to write ckpt file size in elongated file.\n");
  }
  
  // Now let us turn to the actual encoding. 

  // We initialize the matrix. For now we will take a Cauchy matrix with X the first topo->groupSize elements of GF(2^w) and
  // Y the next topo->groupSize elements of GF(2^w). We build the matrix using GF(2^w).

  
  int *matrix, *bitmatrix;
  int **schedule;
  matrix = (int *)malloc(topo->groupSize*topo->groupSize*sizeof(int));
  int j;
  int X[topo->groupSize], Y[topo->groupSize];
  /* for(j=0;j<topo->groupSize;j++){
    X[j]=j;
  }
  for(j=topo->groupSize;j<2*topo->groupSize;j++){
    Y[j%topo->groupSize]=j;
  }  */
  X[0]=1;
  X[1]=2;
  X[2]=244;
  X[3]=247;
  Y[0]=0;
  Y[1]=3;
  Y[2]=245;
  Y[3]=246;
  for(i=0;i<topo->groupSize;i++){
    for(j=0;j<topo->groupSize;j++){
      matrix[i*topo->groupSize+j]=galois_single_divide(1,(X[i]^Y[j]),w);
    }
  }
  //jerasure_print_matrix(matrix,4,4,8);
  bitmatrix = jerasure_matrix_to_bitmatrix(topo->groupSize,topo->groupSize,w,matrix);
  schedule = jerasure_smart_bitmatrix_to_schedule(topo->groupSize,topo->groupSize,w,bitmatrix);
  
  lseek(fd_m, 0, SEEK_SET); 
 
  char *dataBlock; // All processes will need to store temporarily a piece of raw ckpt image.
  char ***dataBlocks;
  char *dataBlocks_region; //have all data blocks stored contiguously in memory for easier handling
  char ***encodedBlocks; // Group rank 0 is going to temporarily store all the pieces of encoded files before sending them to 
                         // respective processes
  char *encodedBlock; // Every node is going to keep receiving from group rank 0 pieces of the final encoded files
  char *encodedBlocks_region; //have all encoded blocks stored contiguously in memory for easier handling


  int num_process = 4; //number of "size" elements to process per rank
  int batch_size = size*num_process;

  dataBlock = (char*)malloc(batch_size);
  dataBlocks = (char***)malloc(num_process*sizeof(char **));
  dataBlocks_region = (char *)malloc(topo->groupSize*batch_size);
  encodedBlock = (char*)malloc(batch_size);
  encodedBlocks = (char***)malloc(num_process*sizeof(char **));
  encodedBlocks_region = (char *)malloc(topo->groupSize*batch_size);
  for (int i = 0; i < num_process; i++){
    dataBlocks[i] = (char **)malloc(topo->groupSize*sizeof(char *));
    encodedBlocks[i] = (char **)malloc(topo->groupSize*sizeof(char *));
    for (int j = 0; j < topo->groupSize; j++){
      //properly assign data blocks
      dataBlocks[i][j] = &dataBlocks_region[i*size+j*batch_size];
      encodedBlocks[i][j] = &encodedBlocks_region[i*size+j*batch_size];
    }
  }

  MD5_CTX context_og, context_enc;
  unsigned char digest_og[16], digest_enc[16];
  MD5_Init(&context_og);
  MD5_Init(&context_enc);

  int toProcess[topo->groupSize];
  int k, num;
  int pos = 0;
  double time_pre = 0, time_comp = 0, time_post = 0;
  double wtime;
  //wtime = MPI_Wtime();
  while(pos<final_size){

    wtime = MPI_Wtime();
    //distribute blocks across all ranks from group    
    for(i = 0; i < topo->groupSize; i++){
      toProcess[i] = (pos+batch_size <= final_size) ? 
                            batch_size : final_size-pos;
      pos += toProcess[i];
      if (toProcess[i] == 0) continue;
      Util::readAll(fd_m, dataBlock, toProcess[i]);
      //communication needs to always be with batch_size elements,
      //otherwise the contiguous memory region assignments
      //are incorrect
      MPI_Gather(dataBlock, batch_size, MPI_CHAR,
        dataBlocks_region, batch_size, MPI_CHAR, i, topo->groupComm);
      num = toProcess[i]/size;
      for (k = 0; k < num; k++) {
        MD5_Update(&context_og,&dataBlock[size*k],size);
      }
    }

    time_pre += (MPI_Wtime() - wtime);

    wtime = MPI_Wtime();
    // Now we do the encoding, if we have data
    num = toProcess[topo->groupRank]/size;
    for (k = 0; k < num; k++) {
      jerasure_schedule_encode(topo->groupSize,topo->groupSize,w,schedule,dataBlocks[k],encodedBlocks[k],size,packetSize);
    }
    time_comp += (MPI_Wtime() - wtime);

    wtime = MPI_Wtime();

    for (i = 0; i < topo->groupSize; i++){
      if (toProcess[i] == 0) break;
      MPI_Scatter(encodedBlocks_region, batch_size, MPI_CHAR,
        encodedBlock, batch_size, MPI_CHAR, i, topo->groupComm);
      Util::writeAll(fd_e,encodedBlock,toProcess[i]);
      num = toProcess[i]/size;
      for (k = 0; k < num; k++) {
        MD5_Update(&context_enc,&encodedBlock[size*k],size);
      }
    }
    time_post += (MPI_Wtime() - wtime);
    MPI_Barrier(topo->groupComm);
  }

  //printf("encoding time: %.3f\n", MPI_Wtime()-wtime);
  printf("pre:%.3f, comp:%.3f, post:%.3f\n", time_pre, time_comp, time_post);
  fflush(stdout);

  free(matrix);
  free(bitmatrix);
  free(schedule);
  free(dataBlock);
  free(dataBlocks);
  free(encodedBlocks);
  free(dataBlocks_region);
  free(encodedBlocks_region);

  int fd_e_chksum = open(encodedChksum.c_str(), O_CREAT | O_TRUNC | O_WRONLY, S_IRUSR | S_IWUSR);
  JASSERT(fd_e_chksum != -1);

  int fd_m_chksum = open(ckptChksum.c_str(), O_CREAT | O_TRUNC | O_WRONLY, S_IRUSR | S_IWUSR);
  JASSERT(fd_m_chksum != -1);

  MD5_Final(digest_enc, &context_enc); 
  Util::writeAll(fd_e_chksum, digest_enc, 16);
  

  MD5_Final(digest_og, &context_og);
  Util::writeAll(fd_m_chksum, digest_og, 16);

  close(fd_m);
  close(fd_e);
  close(fd_e_chksum);
  close(fd_m_chksum);
}

void UtilsMPI::performRSDecoding(string filename, string filenameEncoded, Topology* topo, int *to_recover, int *erasures, int total_succes_raw, int *survivors){
  int w = 8;
  int packetSize = 16*1024*sizeof(long);
  int size = w*packetSize;

  uint64_t post_start;
  if(topo->groupRank==0){
    post_start = getRealCurrTime();
    printf("Performing RS decoding...\n");
    fflush(stdout);
  }

  // We first set the matrix which we will need to take the inverse of
  int *matrix,*inverse_matrix, *bitmatrix;
  if(topo->groupRank==0){
    matrix = (int *)malloc(topo->groupSize*topo->groupSize*sizeof(int));
    int j;
    int X[topo->groupSize], Y[topo->groupSize];
    if((topo->groupSize == 4) && (w==8)){
      X[0]=1;
      X[1]=2;
      X[2]=244;
      X[3]=247;
      Y[0]=0;
      Y[1]=3;
      Y[2]=245;
      Y[3]=246;
    } else{
      for(j=0;j<topo->groupSize;j++){
        X[j]=j;
      }
      for(j=topo->groupSize;j<2*topo->groupSize;j++){
        Y[j%topo->groupSize]=j;
      } 
    }
    for(int i=0;i<topo->groupSize;i++){
      for(j=0;j<topo->groupSize;j++){
        matrix[i*topo->groupSize+j]=galois_single_divide(1,(X[i]^Y[j]),w);
      }
    }
    bitmatrix = jerasure_matrix_to_bitmatrix(topo->groupSize,topo->groupSize,w,matrix);
  }

  // At this point we have to proceed similarly to as we did in the encoding procedure, with the exception that to do the 
  // computations we have to retrieve the survivor files, and we only need to send the resulting recovered files to
  // the processes in to_recover

  // We need to know the ckpt file sizes, which are all the same. So we can get the size from just one process, provided that
  // we are cautious not to select a process which has been corrupted
  off_t ckptFileSize;
  if(survivors[0]<topo->groupSize){
    if(topo->groupRank==survivors[0]){
      int fd_m = open(filename.c_str(), O_RDONLY);
      JASSERT(fd_m != -1);
      struct stat sb;
      JASSERT(fstat(fd_m, &sb) == 0);
      ckptFileSize = sb.st_size;
      close(fd_m);
    }
    MPI_Bcast(&ckptFileSize,sizeof(off_t),MPI_CHAR,survivors[0],topo->groupComm);
  }else if(survivors[0]>=topo->groupSize){
    if(topo->groupRank==survivors[0]%topo->groupSize){
      string filename_encoded = filename + "_encoded";
      int fd_m = open(filename_encoded.c_str(), O_RDONLY);
      JASSERT(fd_m != -1);
      struct stat sb;
      JASSERT(fstat(fd_m, &sb) == 0);
      ckptFileSize = sb.st_size;
      close(fd_m);
    }
    MPI_Bcast(&ckptFileSize,sizeof(off_t),MPI_CHAR,survivors[0]-topo->groupSize,topo->groupComm);
  }

  char *dataBlock; // All processes will need to store temporarily a piece of raw ckpt image.
  char **dataBlocks;
  char **encodedBlocks; // Group rank 0 is going to temporarily store all the pieces of encoded files before sending them to 
                         // respective processes
  char *encodedBlock; // Every node is going to keep receiving from group rank 0 pieces of the final encoded files

  if(topo->groupRank==0){
    dataBlock = (char*)malloc(size);
    dataBlocks = (char**)malloc(topo->groupSize*sizeof(char *));
    encodedBlock = (char*)malloc(size);
    encodedBlocks = (char**)malloc(topo->groupSize*sizeof(char *));
    for(int i = 0;i<topo->groupSize;i++){
      dataBlocks[i] = (char*)malloc(size);
      encodedBlocks[i] = (char*)malloc(size);
    }
  }else{
    dataBlock = (char*)malloc(size);
    encodedBlock = (char*)malloc(size);
  }
  
  int num_to_recover = topo->groupSize-total_succes_raw;

  string encodedChksum = filenameEncoded + "_md5chksum";
  int fd_m,fd_e;

  for(int i=0;i<topo->groupSize;i++){
    if(survivors[i]<topo->groupSize){
      if(topo->groupRank==survivors[i]){
        fd_m = open(filename.c_str(), O_RDONLY);
        JASSERT(fd_m != -1);
      }
    }else if (survivors[i]>=topo->groupSize){
      if(topo->groupRank==survivors[i]%topo->groupSize){
        fd_e = open(filenameEncoded.c_str(), O_RDONLY);
        JASSERT(fd_e != -1);
      }
    }
  }

  for(int i=0;i<num_to_recover;i++){
    if(topo->groupRank==to_recover[i]){
      fd_m = open(filename.c_str(), O_CREAT | O_TRUNC | O_WRONLY, S_IRUSR | S_IWUSR);
      if(truncate(filename.c_str(),0) == -1){
        JASSERT(0).Text("Error with truncation on ckpt image.\n");
      }
      close(fd_m);
      fd_m = open(filename.c_str(), O_CREAT | O_TRUNC | O_WRONLY, S_IRUSR | S_IWUSR);
    }
  }

  int pos = 0;
  while(pos<ckptFileSize){
    if(topo->groupRank==0){
      // We have to iteratively acquire the data from survivors[i]
      for(int i=0;i<topo->groupSize;i++){
        if(survivors[i]<topo->groupSize){
          if(survivors[i]==0){
            Util::readAll(fd_m,dataBlocks[0],size);
          } else{
            MPI_Recv(dataBlocks[survivors[i]],size,MPI_CHAR,survivors[i],0,topo->groupComm,MPI_STATUS_IGNORE);
          }
        }else if(survivors[i]>=topo->groupSize){
          if(survivors[i]%topo->groupSize==0){
            Util::readAll(fd_e,encodedBlocks[0],size);
          } else{
            MPI_Recv(encodedBlocks[survivors[i]%topo->groupSize],size,MPI_CHAR,survivors[i]%topo->groupSize,0,topo->groupComm,MPI_STATUS_IGNORE);
          }
        }
      }
    } else{
      for(int i=0;i<topo->groupSize;i++)
      {
        if(survivors[i]<topo->groupSize && survivors[i]!=0 && topo->groupRank==survivors[i]){
          Util::readAll(fd_m,dataBlock,size);
          MPI_Send(dataBlock,size,MPI_CHAR,0,0,topo->groupComm);
        } else if(survivors[i]>= topo->groupSize && survivors[i]%topo->groupSize!=0 && topo->groupRank==survivors[i]%topo->groupSize){
          Util::readAll(fd_e,dataBlock,size);
          MPI_Send(dataBlock,size,MPI_CHAR,0,0,topo->groupComm);
        }
      }
    }

    // Now we decode
    if(topo->groupRank==0){
      jerasure_schedule_decode_lazy(topo->groupSize,topo->groupSize,w,bitmatrix,erasures,dataBlocks,encodedBlocks,size,packetSize,1);
    }

    // Now we have to send back the pieces of recovered ckpt files to their respective processes, with the caution that 
    // process group Rank 0 can be one of the processes in to_recover and there is no need to send in that.
    for(int i=0;i<num_to_recover;i++){
      if(to_recover[i]==0){
        if(topo->groupRank==0){
          Util::writeAll(fd_m,dataBlocks[0],size);
        }
      }else{
        if(topo->groupRank==0){
          MPI_Send(dataBlocks[to_recover[i]],size,MPI_CHAR,to_recover[i],0,topo->groupComm);
        }else{
          if(topo->groupRank==to_recover[i]){
            MPI_Recv(dataBlock,size,MPI_CHAR,0,0,topo->groupComm,MPI_STATUS_IGNORE);
            Util::writeAll(fd_m,dataBlock,size);
          }
        }
      }
    }
    MPI_Barrier(topo->groupComm);
    pos +=size;
  }
  
  if(topo->groupRank==0){
    free(matrix);
    free(bitmatrix);
    free(dataBlock);
    for(int i = 0;i<topo->groupSize;i++){
      free(dataBlocks[i]);
      free(encodedBlocks[i]);
    }
    free(dataBlocks);
    free(encodedBlocks);
  }else{
    free(encodedBlock);
  }

  for(int i=0;i<topo->groupSize;i++){
    if(survivors[i]<topo->groupSize){
      if(topo->groupRank==survivors[i]){
        close(fd_m);
      }
    }else if (survivors[i]>=topo->groupSize){
      if(topo->groupRank==survivors[i]%topo->groupSize){
        close(fd_e);
      }
    }
  }

  for(int i=0;i<num_to_recover;i++){
    if(topo->groupRank==to_recover[i]){
      close(fd_m);
    }
  }

  if(topo->groupRank==0){
    uint64_t post_time = (getRealCurrTime() - post_start)/1000;
    double post_time_sec = ((double)post_time)/1000;
    if(UtilsMPI::instance().getRank() == 0){
      printf("RS decoding took %.3f seconds.\n", post_time_sec);
    }
  }

  MPI_Barrier(topo->groupComm);
}

void UtilsMPI::convertMatrixToBitmatrix(int *matrix, int *bitmatrix, int n, int w){
  // We assume the matrices have been allocated with the correct sizes
  int i,j,k,l;
  int matrix_val;
  int power;
  for(i=0;i<n;i++){ // Iterates vertically through the n x n original matrix (still not bit)
    for(j=0;j<n;j++){ // Iterates horizontally through the n x n original matrix (still not bit)
      matrix_val = matrix[i*n+j];
      for(l=0;l<w;l++){ // At each column of the submatrix (bit) for the matrix[i][j] element, we have the bit representation of matrix_val*2^(l)
        // We obtain the bit representation of matrix_val*2^(l)
        power = 1;
        for(k=0;k<w;k++){
          bitmatrix[i*w*w*n+j*w+k*w*n+l] = ((matrix_val & power) == 0) ? 0 : 1;
          power *= 2;
        }
        matrix_val = galois_single_multiply(matrix_val,2,w);
      }
    }
  }
  /* for(int q=0;q<n*w;q++){
    for(int r=0;r<n*w;r++){
      printf("%d ", bitmatrix[q*n*w+r]);
    }
    printf("\n");
  }
  fflush(stdout); */
}
  
char*
findCkptFilename(string ckpt_dir, string patternEnd){
  int file_found = 0;  
  DIR *dir;
  struct dirent *entry;
  char *result = (char *)malloc(1024);
  dir = opendir(ckpt_dir.c_str());
  if (dir == NULL) {
    printf("  Checkpoint directory %s not found.\n", ckpt_dir.c_str());
    return 0;
  }
  while ((entry = readdir(dir)) != NULL){
    if(Util::strStartsWith(entry->d_name, "ckpt") &&
            Util::strEndsWith(entry->d_name, patternEnd.c_str())){
      sprintf(result, "%s/%s", ckpt_dir.c_str(), entry->d_name);
      file_found = 1;
      break;
    }
  }
  if(!file_found) free(result);

  return (file_found) ? result : NULL;
}


/**
 *
 * Fake reading of ProcessInfo data for accessing later data
 * to avoid modifying the underlying process.
 *
 */
void
dummySerialize(jalib::JBinarySerializer &o){
  uint64_t dummy64;
  uint32_t dummy32;
  pid_t dummyPid;
  UniquePid dummyUPid;
  string dummyStr;
  map<pid_t, UniquePid> dummyMap;

  JSERIALIZE_ASSERT_POINT("ProcessInfo:");

  o & dummy32;
  o & dummy32 & dummyPid & dummyPid & dummyPid & dummyPid & dummyPid & dummy32;
  o & dummyStr & dummyStr & dummyStr & dummyStr & dummyStr;
  o & dummyUPid & dummyUPid;
  o & dummy64 & dummy64
    & dummy64 & dummy64;
  o & dummyUPid & dummy32 & dummy32 & dummy32 & dummy32;
  o & dummy64 & dummy64 & dummy64;
  o & dummy64 & dummy64 & dummy64 & dummy64 & dummy64;
  int i;
  for (i = 0; i <= CKPT_GLOBAL; i++){
    o & dummyStr & dummyStr & dummyStr;
  }
  o & dummyMap;

  JSERIALIZE_ASSERT_POINT("EOF");
}

int
readDmtcpHeader(int fd){
  const size_t len = strlen(DMTCP_FILE_HEADER);
  
  char *buff = (char *)malloc(len);
  if(Util::readAll(fd, buff, len) != len){
    free(buff);
    return 0;
  }
  free(buff);

  jalib::JBinarySerializeReaderRaw rdr("", fd);

  //careful now....
  dummySerialize(rdr);

  size_t numRead = len + rdr.bytes();
  
  // We must read in multiple of PAGE_SIZE
  const ssize_t pagesize = Util::pageSize();
  ssize_t remaining = pagesize - (numRead % pagesize);
  char buf[remaining];

  return (Util::readAll(fd, buf, remaining) == remaining);
}

int
readMtcpHeader(int fd){
  MtcpHeader mtcpHdr;
  return (Util::readAll(fd, &mtcpHdr, sizeof(mtcpHdr)) == sizeof(mtcpHdr)); 
}

int
UtilsMPI::isCkptValid(const char *filename){
  MD5_CTX context;
  unsigned char digest[16], chksum_read[16];
  string ckptFile = filename;
  string ckptChecksum = ckptFile + "_md5chksum";

  int fd = open(ckptFile.c_str(), O_RDONLY);
  int fd_chksum = open(ckptChecksum.c_str(), O_RDONLY);
  
  if (fd == -1 || fd_chksum == -1){
    //treat the absence or lack of ablity to open
    //either of the two files as a failure
    printf("  Checkpoint or checksum file missing.\n");
    if(fd != -1) close(fd);
    if(fd_chksum != -1) close(fd_chksum);
    return 0;
  }

  //read checksum file
  if(Util::readAll(fd_chksum, chksum_read, 16) != 16){
    printf("  Checksum file is smaller than 16 bytes.\n");
    close(fd); close (fd_chksum);
    return 0;
  }

  //ignore DMTCP header
  if(!readDmtcpHeader(fd)){
    printf("  Error reading DMTCP header.\n");
    close(fd); close (fd_chksum);
    return 0;
  }

  //ignore MTCP header
  if(!readMtcpHeader(fd)){
    printf("  Error reading MTCP header.\n");
    close(fd); close (fd_chksum);
    return 0;
  }

  //read checkpoint file as pure data just to verify checksum
  Area area;
  size_t END_OF_CKPT = -1;
  ssize_t bytes_read;
  int num_updates = 0;
  MD5_Init(&context);
  char *addr_tmp = (char *)malloc(MEM_TMP_SIZE);
  while (Util::readAll(fd, &area, sizeof(area)) == sizeof(area)){
    //if -1 size found, we are finished
    if (area.size == END_OF_CKPT) break;

    //update context with area info
    MD5_Update(&context, &area, sizeof(area));

    //printf("Reading region of %lu bytes...\n", area.size);
    //fflush(stdout);

    //printf("area name: %s\n", area.name);


    //handle non rwx and anonymous pages
    if (area.prot == 0 || (area.name[0] == '\0' &&
          ((area.flags & MAP_ANONYMOUS) != 0) &&
          ((area.flags & MAP_PRIVATE) != 0))){
      if(area.properties == DMTCP_ZERO_PAGE) {
        //zero page, skip
        continue;
      }
    }

    
    if ((area.properties & DMTCP_SKIP_WRITING_TEXT_SEGMENTS) &&
          (area.prot & PROT_EXEC)){
       //skip text segment if applicable
       continue;
    }
   

    uint64_t toRead = area.size;
    while (toRead > 0){
      uint64_t chunkSize = (toRead > MEM_TMP_SIZE) ?
                           MEM_TMP_SIZE : toRead;

      if((bytes_read = Util::readAll(fd, addr_tmp, chunkSize)) != (ssize_t)chunkSize){
        if(bytes_read == -1) {
          printf("  Error reading memory region: %s\n", strerror(errno));
        }
        else {
          printf("  Expected memory region of %lu bytes, got only %ld.\n", area.size, bytes_read);
        }
        close(fd); close (fd_chksum);
        free(addr_tmp);
        return 0;
      }
      MD5_Update(&context, addr_tmp, chunkSize);
      toRead -= chunkSize;
      num_updates++;
    }

  }

  free(addr_tmp);

  if(area.size != END_OF_CKPT){
    printf("  Checkpoint file did not finish as expected.\n");
    close(fd); close (fd_chksum);
    return 0;
  }

  MD5_Final(digest, &context);

  //printf("Number of updates: %d\n", num_updates);
  printf("Rank %d: computed checksum: ", _rank);
  for(int i = 0; i < 16; i++){
    printf("%x", digest[i]);
  }
  printf("\nRank %d: read checksum: ", _rank);
  for(int i = 0; i < 16; i++){
    printf("%x", chksum_read[i]);
  }
  printf("\n");
 
  if(strncmp((char *)digest, (char *)chksum_read, 16) != 0){
    printf("  Rank %d: Computed checksum does not match the checksum file contents.\n", _rank);
    close(fd); close (fd_chksum);
    return 0;
  }

  close(fd);
  close(fd_chksum);
  return 1;
}

int
UtilsMPI::isEncodedCkptValid(const char *filename){
  unsigned char digest[16], chksum_read[16];
  string encodedCkpt = filename;
  string encodedChecksum = encodedCkpt + "_md5chksum";

  int fd_m = open(encodedCkpt.c_str(), O_RDONLY);
  int fd_m_chksum = open(encodedChecksum.c_str(), O_RDONLY);
  
  if (fd_m == -1 || fd_m_chksum == -1){
    //treat the absence or lack of ablity to open
    //either of the two files as a failure
    printf("  Checkpoint or checksum file missing.\n");
    if(fd_m != -1) close(fd_m);
    if(fd_m_chksum != -1) close(fd_m_chksum);
    return 0;
  }

  //read checksum file
  if(Util::readAll(fd_m_chksum, chksum_read, 16) != 16){
    printf("  Checksum file is smaller than 16 bytes.\n");
    close(fd_m); close (fd_m_chksum);
    return 0;
  }

  MD5_CTX context;
  Area area;
  size_t END_OF_CKPT = -1;
  ssize_t bytes_read;
  int chunkSize = MEM_TMP_SIZE;
  char *addr_tmp = (char *)malloc(chunkSize);
  MD5_Init(&context);


  int pos = 0;
  int cont = true;
  while(cont){
    if((bytes_read = Util::readAll(fd_m, addr_tmp, chunkSize)) != (ssize_t)chunkSize){
        cont = false;
        if(bytes_read == -1) {
          printf("Error reading encoded memory region: %s\n", strerror(errno));
          fflush(stdout);
        }
    }
    MD5_Update(&context,addr_tmp,bytes_read);
    pos += chunkSize;
  }
  MD5_Final(digest, &context);
  close(fd_m);
  close(fd_m_chksum);

  printf("Rank %d: computed encoded checksum: ", _rank);
  for(int i = 0; i < 16; i++){
    printf("%x", digest[i]);
  }
  printf("\nRank %d: read encoded checksum: ", _rank);
  for(int i = 0; i < 16; i++){
    printf("%x", chksum_read[i]);
  }
  printf("\n");
 
  if(strncmp((char *)digest, (char *)chksum_read, 16) != 0){
    printf("  Rank %d: Computed encoded checksum does not match the encoded checksum file contents.\n", _rank);
    close(fd_m); close (fd_m_chksum);
    return 0;
  }

  close(fd_m);
  close(fd_m_chksum);
  return 1;
}

/**
 *
 * Asserting anything in this function and its 
 * subfunctions is a bad idea, since crashing the
 * application here defeats its purpose, as we 
 * want to choose alternative checkpoint levels 
 * if the the checkpoint being checked has any 
 * issues.
 *
 */
int
UtilsMPI::checkCkptValid(int ckpt_type, string ckpt_dir, Topology *topo){
  string real_dir = ckpt_dir;
  char *rankchar = (char *)malloc(32);
  sprintf(rankchar, "/ckpt_rank_%d/", _rank);
  real_dir += rankchar;
  int success = 0, allsuccess, partnerSuccess;
  string ckptFilename;
  if(ckpt_type != CKPT_SOLOMON){
    try {
      char *filename = findCkptFilename(real_dir, ".dmtcp");
      if(filename != NULL && isCkptValid(filename)){
          success = 1;
          if(ckpt_type == CKPT_PARTNER){
            //assist partner if necessary
            int partnerCkptValid = 0;
            char *partnerCkpt = NULL;
            MPI_Sendrecv(&success, 1, MPI_INT, topo->right, 0, 
              &partnerSuccess, 1, MPI_INT, topo->left, 0, topo->groupComm, MPI_STATUS_IGNORE);
            if(!partnerSuccess){
              //check if our partner ckpt is valid
              printf("Rank %d | Group rank %d: Partner group rank %d requested assist with partner copy.\n", _rank, topo->groupRank, topo->left);
              fflush(stdout);
              partnerCkpt = findCkptFilename(real_dir, ".dmtcp_partner");
              if(partnerCkpt != NULL){
                partnerCkptValid = isCkptValid(partnerCkpt);
              }
              if(!partnerCkptValid){
                printf("Rank %d | Group rank %d: Partner copy ckpt not valid.\n", _rank, topo->groupRank);
                fflush(stdout);
              }
            }
              
            //send help response to partner
            int myCkptValid, recoverable;
            MPI_Sendrecv(&partnerCkptValid, 1, MPI_INT, topo->left, 0,
                  &myCkptValid, 1, MPI_INT, topo->right, 0, topo->groupComm, MPI_STATUS_IGNORE);
            //we know our checkpoint is valid
            myCkptValid = 1;
            //only enter recovery process if all ranks can recover
            MPI_Allreduce(&myCkptValid, &recoverable, 1, MPI_INT, MPI_MIN, topo->groupComm);
            if(recoverable && !partnerSuccess && partnerCkptValid){
              //assist
              if(!assistPartnerCopy(partnerCkpt, topo)){
                printf("Rank %d | Group rank %d: error while assisting group rank %d with partner copy\n", _rank, topo->groupRank, topo->left);
                fflush(stdout);
              }
            }
            if(partnerCkpt != NULL) free(partnerCkpt);
          }
      }
      else {
        if(ckpt_type == CKPT_PARTNER){
          //recover from partner if possible
          printf("Rank %d | Group rank %d: Requesting recovery with partner group rank %d\n", _rank, topo->groupRank, topo->right);
          MPI_Sendrecv(&success, 1, MPI_INT, topo->right, 0, 
              &partnerSuccess, 1, MPI_INT, topo->left, 0, topo->groupComm, MPI_STATUS_IGNORE);
          int partnerCkptValid = 0, myCkptValid = 0;
          char *partnerCkpt = NULL;
          int recoverable = 0;
          if(!partnerSuccess){
            printf("Rank %d | Group rank %d: Partner group rank %d requested assist with partner copy.\n", _rank, topo->groupRank, topo->left);
            fflush(stdout);
            partnerCkpt = findCkptFilename(real_dir, ".dmtcp_partner");
            if(partnerCkpt != NULL){
              partnerCkptValid = isCkptValid(partnerCkpt);
            }
            if(!partnerCkptValid){
              printf("Rank %d | Group rank %d: Partner copy ckpt not valid.\n", _rank, topo->groupRank);
              fflush(stdout);
            }
          }
          //send/recv help response to partner
          MPI_Sendrecv(&partnerCkptValid, 1, MPI_INT, topo->left, 0,
                  &myCkptValid, 1, MPI_INT, topo->right, 0, topo->groupComm, MPI_STATUS_IGNORE);
          //only enter recovery process if all ranks can recover
          MPI_Allreduce(&myCkptValid, &recoverable, 1, MPI_INT, MPI_MIN, topo->groupComm);
          if(recoverable){
            int ret;
            string fileCkpt;
            if(filename == NULL){
              //if file not found, construct ckpt file
              filename = (char *)malloc(1024);
              sprintf(filename, "%s/ckptzRestored.dmtcp", real_dir.c_str());
            }
            if(topo->groupRank % 2 == 0 ){
              ret = recoverFromPartnerCopy(filename, topo);
              if(!partnerSuccess){
                assistPartnerCopy(partnerCkpt, topo);
              }
            }
            else {
              if(!partnerSuccess){
                assistPartnerCopy(partnerCkpt, topo);
              }
              ret = recoverFromPartnerCopy(filename, topo);
            }
            //verify ckpt
            if (ret){
              success = isCkptValid(filename);
            }
            if(!ret || !success){
              printf("Rank %d | Group rank %d: Partner ckpt transmission was not successful or ckpt is corrupt.\n", _rank, topo->groupRank);
              fflush(stdout);
            }
          }
          if(partnerCkpt != NULL) free(partnerCkpt);
        }
      }
      if(filename != NULL) free(filename);
    }
    catch (std::exception &e){ //survive unexpected exceptions and assume failure
      printf("Rank %d | Group rank %d: An exception occurred(%s).\n", _rank, topo->groupRank, e.what());
    }
  }else if(ckpt_type==CKPT_SOLOMON){
      char *fileName, *fileName_encoded;
      string filename, filename_encoded;
      int encoded_success=0;
      fileName = findCkptFilename(real_dir, ".dmtcp");
      fileName_encoded = findCkptFilename(real_dir, ".dmtcp_encoded");
      if(fileName!=NULL){
        filename = fileName;
      }else{
        success=0;
      }
      if(fileName_encoded!=NULL){
        filename_encoded = fileName_encoded;
        if(isEncodedCkptValid(filename_encoded.c_str())){
          encoded_success = 1;
        }
      }
      if(fileName != NULL){
        if(isEncodedCkptValid(filename.c_str())){
          success=1;
        }
      }

      int success_array[topo->groupSize], encoded_success_array[topo->groupSize];
      int total_success_raw=0,total_success_encoded=0;
      MPI_Allgather(&success,1,MPI_INT,success_array,1,MPI_INT,topo->groupComm);
      MPI_Allgather(&encoded_success,1,MPI_INT,encoded_success_array,1,MPI_INT,topo->groupComm);

      // We have to determine whether we can recover from this number of failures 
      int i;
      for(i=0;i<topo->groupSize;i++){
        total_success_raw += success_array[i];
        total_success_encoded += encoded_success_array[i];
      }
      if(total_success_raw + total_success_encoded < topo->groupSize){
        return 0;
      }

      // If all the raw ckpt file images, we can proceed without any decoding
      if(total_success_raw == topo->groupSize){
        return 1;
      }

      // If we get here, we can recover from the ckpt raw and encoded files. We have to identify which ckpt raw files we need to
      // recover.

      int to_recover[topo->groupSize-total_success_raw];
      int erasures[2*topo->groupSize];
      int survivors[topo->groupSize]; // We just need topo->groupSize correct files to recover all the ckpt images. We select
                                      // all the unencoded correct ones and, if necesssary, encoded ones.
      int pos_recover = 0, pos_survivor = 0;
      for(i=0;i<topo->groupSize;i++){
        if(success_array[i]==0){
          to_recover[pos_recover]=i;
          pos_recover++;
        }else{
          survivors[pos_survivor]=i;
          pos_survivor++;
        }
      }
      pos_recover = 0;
      for(i=0;i<topo->groupSize;i++){
        if(success_array[i]==0){
          erasures[pos_recover] = i;
          pos_recover++;
        }
      }
      for(i=0;i<topo->groupSize;i++){
        if(encoded_success_array[i]==0){
          erasures[pos_recover] = i+topo->groupSize;
          pos_recover++;
        }
      }
      erasures[pos_recover] = -1;

      // We add the necessary encoded ckpt files. We will denote them by topo->groupSize + i, where i is the process to which they belong.
      for(i=0;i<topo->groupSize;i++){
        if(pos_survivor==topo->groupSize){
          break;
        }
        if(encoded_success_array[i]==1){
          survivors[pos_survivor]=topo->groupSize+i;
          pos_survivor++;
        }
      }

      if (topo->groupRank == 0) {
      for(i = 0; i < pos_recover; i++){
        printf("%d", to_recover[i]);
      }
      printf("\n");
      for(i = 0; i < pos_survivor; i++){
        printf("%d", survivors[i]);
      }
      printf("\n");
      }

      // At this point we have already selected the survivor files from which we are going to recover. We would now need to apply
      // RS decoding, via multiplication with the inverse of the adequate matrix.

      if(fileName == NULL){
        filename = real_dir + "ckpt_recovered.dmtcp";
      }
      if(fileName_encoded==NULL){
        filename_encoded = real_dir + "ckpt_encoded_recovered.dmtcp_encoded";
      }

      performRSDecoding(filename,filename_encoded, topo,to_recover,erasures,total_success_raw,survivors);
      return 1;
  }

  printf("Success of rank %d: %d\n", _rank, success);
  fflush(stdout);

  //aggregate operation status from all ranks
  MPI_Allreduce(&success, &allsuccess, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

  free(rankchar);

  // only return success if all ranks are successful
  return allsuccess;
}

string
UtilsMPI::recoverFromCrash(ConfigInfo *cfg){
    RestartInfo *restartInfo = new RestartInfo();
    Topology *topo;
    string target = "";
    int i, found = 1, valid = 0, candidate;

    /*
    for(int i = 0; i <= CKPT_GLOBAL; i++){
      printf("type: %d, dir: %s, time: %ld\n", i, restartInfo->ckptDir[i].c_str(), restartInfo->ckptTime[i]);
    }
    fflush(stdout);
    */

    //read restart information
    restartInfo->readRestartInfo();

    //get system topology
    getSystemTopology(cfg, &topo);

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
        if (_rank == 0){
          printf("Checking checkpoint dir %s...\n", target.c_str());
          fflush(stdout);
        }
        valid = checkCkptValid(candidate, target, topo);
        if(!valid){
          //if not valid, set time to zero to invalidate
          //and loop again
          restartInfo->ckptTime[candidate] = 0;
        }
      }
    }

    JASSERT(found).Text("Restart point not found.");

    if(_rank == 0){
      printf("Selected checkpoint type %d with location %s\n", candidate, target.c_str());
      fflush(stdout);
    }

    return target;
}

int 
UtilsMPI::jerasure_invert_matrix(int *mat, int *inv, int rows, int w)
{
  // This functions has been copied from the Fast Galois Field Arithmetic Library in C/C++ by James S. Plank
  int cols, i, j, k, x, rs2;
  int row_start, tmp, inverse;
 
  cols = rows;

  k = 0;
  for (i = 0; i < rows; i++) {
    for (j = 0; j < cols; j++) {
      inv[k] = (i == j) ? 1 : 0;
      k++;
    }
  }

  /* First -- convert into upper triangular  */
  for (i = 0; i < cols; i++) {
    row_start = cols*i;

    /* Swap rows if we ave a zero i,i element.  If we can't swap, then the 
       matrix was not invertible  */

    if (mat[row_start+i] == 0) { 
      for (j = i+1; j < rows && mat[cols*j+i] == 0; j++) ;
      if (j == rows) return -1;
      rs2 = j*cols;
      for (k = 0; k < cols; k++) {
        tmp = mat[row_start+k];
        mat[row_start+k] = mat[rs2+k];
        mat[rs2+k] = tmp;
        tmp = inv[row_start+k];
        inv[row_start+k] = inv[rs2+k];
        inv[rs2+k] = tmp;
      }
    }
 
    /* Multiply the row by 1/element i,i  */
    tmp = mat[row_start+i];
    if (tmp != 1) {
      inverse = galois_single_divide(1, tmp, w);
      for (j = 0; j < cols; j++) { 
        mat[row_start+j] = galois_single_multiply(mat[row_start+j], inverse, w);
        inv[row_start+j] = galois_single_multiply(inv[row_start+j], inverse, w);
      }
    }

    /* Now for each j>i, add A_ji*Ai to Aj  */
    k = row_start+i;
    for (j = i+1; j != cols; j++) {
      k += cols;
      if (mat[k] != 0) {
        if (mat[k] == 1) {
          rs2 = cols*j;
          for (x = 0; x < cols; x++) {
            mat[rs2+x] ^= mat[row_start+x];
            inv[rs2+x] ^= inv[row_start+x];
          }
        } else {
          tmp = mat[k];
          rs2 = cols*j;
          for (x = 0; x < cols; x++) {
            mat[rs2+x] ^= galois_single_multiply(tmp, mat[row_start+x], w);
            inv[rs2+x] ^= galois_single_multiply(tmp, inv[row_start+x], w);
          }
        }
      }
    }
  }

  /* Now the matrix is upper triangular.  Start at the top and multiply down  */

  for (i = rows-1; i >= 0; i--) {
    row_start = i*cols;
    for (j = 0; j < i; j++) {
      rs2 = j*cols;
      if (mat[rs2+i] != 0) {
        tmp = mat[rs2+i];
        mat[rs2+i] = 0; 
        for (k = 0; k < cols; k++) {
          inv[rs2+k] ^= galois_single_multiply(tmp, inv[row_start+k], w);
        }
      }
    }
  }
  return 0;
}

unsigned int UtilsMPI::countSetBits(unsigned char n)
{
    unsigned int count = 0;
    while (n) {
        count += n & 1;
        n >>= 1;
    }
    return count;
}

void UtilsMPI::setMatrixOnes(int* matrix, int w){
  int num_elem = exp2(w);
  int i,j,k;
  int val;
  for(i = 0;i<num_elem;i++){
    for(j=0;j<i;j++){
      val = galois_single_divide(1,i^j,w);
      matrix[i*num_elem+j] = countSetBits(val);
      for(k=1;k<w;k++){
        val = galois_single_multiply(val,2,w);
        matrix[i*num_elem+j] += countSetBits(val);
      }
    }
    matrix[i*num_elem+i] = 0;
    for(j=i+1;j<num_elem;j++){
      val = galois_single_divide(1,i^j,w);
      matrix[i*num_elem+j] = countSetBits(val);
      for(k=1;k<w;k++){
        val = galois_single_multiply(val,2,w);
        matrix[i*num_elem+j] += countSetBits(val);
      }
    }
  }
}

bool isIn(int* vec,int n, int val){
  int i;
  for(i=0;i<n;i++){
    if(val==vec[i]){
      return true;
    }
  }
  return false;
}

void UtilsMPI::findGoodCauchyMatrix(int n, int w){
  int *matrixOnes;
  int *X,*Y;
  int num_elem = exp2(w);

  matrixOnes = (int *)malloc(num_elem*num_elem*sizeof(int));
  setMatrixOnes(matrixOnes,w);

  // The current implementation will only work with n being a power of 2. We follow the algorithm sketched by Plank's paper
  // Therefore, in order to obtain GC(n,n,w) for n being a power of 2 different than 2 itself, we have to iretatively find GC(n/2,n/2,w)

  // As per the paper, we already start with X={1,2} and Y={}
  X = (int*)malloc(n*sizeof(int));
  Y = (int*)malloc(n*sizeof(int));

  X[0]=1;
  X[1]=2;

  // At least we are going to have a first step where we want to find GC(2,2,w)
  int i,j;
  int k;
  int numZeros = 4*w*w+1; // The maximum number of zeros is going to be 4*w*w
  int tmpSum;
  for(i=0;i<num_elem;i++){
    if(isIn(X,2,i)){
      continue;
    }
    for(j=i+1;j<num_elem;j++){
      if(isIn(X,2,j)){
        continue;
      }
      tmpSum = 0;
      tmpSum += matrixOnes[i*num_elem+1];
      tmpSum += matrixOnes[i*num_elem+2];
      tmpSum += matrixOnes[j*num_elem+1];
      tmpSum += matrixOnes[j*num_elem+2];
      if(tmpSum<numZeros){
        Y[0] = i;
        Y[1] = j;
        numZeros = tmpSum;
      }
    }
  }

  // Now we have to find GC(4,4,w). At this point we have X={1,2} and Y={k1,k2} with some k1,k2 in GF(2^w)

  int l,m;
  int p,q;

  numZeros = 16*w*w+1; // The maximum number of zeros is going to be 4*w*w

  for(i=0;i<num_elem;i++) {
    if(isIn(X,2,i)||isIn(Y,2,i)){
      continue;
    }
    for(j=i+1;j<num_elem;j++){
      if(isIn(X,2,j)||isIn(Y,2,j)){
        continue;
      }
      for(l=0;l<num_elem;l++){
        if(isIn(X,2,l) || isIn(Y,2,l) || l==i || l==j){
          continue;
        }
        for(m=l+1;m<num_elem;m++){
          if(isIn(X,2,m) || isIn(Y,2,m) || m==i || m==j){
            continue;
          }
          tmpSum = 0;
          tmpSum += matrixOnes[Y[0]*num_elem+1]; tmpSum += matrixOnes[Y[0]*num_elem+2]; tmpSum += matrixOnes[Y[0]*num_elem+i]; tmpSum += matrixOnes[Y[0]*num_elem+j];
          tmpSum += matrixOnes[Y[1]*num_elem+1]; tmpSum += matrixOnes[Y[1]*num_elem+2]; tmpSum += matrixOnes[Y[1]*num_elem+i]; tmpSum += matrixOnes[Y[1]*num_elem+j];
          tmpSum += matrixOnes[l*num_elem+1]; tmpSum += matrixOnes[l*num_elem+2]; tmpSum += matrixOnes[l*num_elem+i]; tmpSum += matrixOnes[l*num_elem+j];
          tmpSum += matrixOnes[m*num_elem+1]; tmpSum += matrixOnes[m*num_elem+2]; tmpSum += matrixOnes[m*num_elem+i]; tmpSum += matrixOnes[m*num_elem+j];
          if(tmpSum<numZeros){
            X[2] = i;
            X[3] = j;
            Y[2] = l;
            Y[3] = m;
            numZeros = tmpSum;
          }
        }
      }
    }
  }

  printf("X: ");
  for(i =0;i<4;i++){
    printf("%d ",X[i]);
  }
  printf("\n");
  printf("Y: ");
  for(i =0;i<4;i++){
    printf("%d ",Y[i]);
  }
  printf("\n");
  printf("Sum: %d\n",numZeros);
  fflush(stdout);

  free(X);
  free(Y);
  free(matrixOnes);
}

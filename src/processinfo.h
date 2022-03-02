/****************************************************************************
 *   Copyright (C) 2006-2008 by Jason Ansel, Kapil Arya, and Gene Cooperman *
 *   jansel@csail.mit.edu, kapil@ccs.neu.edu, gene@ccs.neu.edu              *
 *                                                                          *
 *  This file is part of DMTCP.                                             *
 *                                                                          *
 *  DMTCP is free software: you can redistribute it and/or                  *
 *  modify it under the terms of the GNU Lesser General Public License as   *
 *  published by the Free Software Foundation, either version 3 of the      *
 *  License, or (at your option) any later version.                         *
 *                                                                          *
 *  DMTCP is distributed in the hope that it will be useful,                *
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of          *
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the           *
 *  GNU Lesser General Public License for more details.                     *
 *                                                                          *
 *  You should have received a copy of the GNU Lesser General Public        *
 *  License along with DMTCP:dmtcp/src.  If not, see                        *
 *  <http://www.gnu.org/licenses/>.                                         *
 ****************************************************************************/

#ifndef PROCESS_INFO_H
#define PROCESS_INFO_H

#include <sys/types.h>
#include "../jalib/jalloc.h"
#include "uniquepid.h"
#include "constants.h"

#define MB                 1024 * 1024
#define RESTORE_STACK_SIZE 5 * MB
#define RESTORE_MEM_SIZE   5 * MB
#define RESTORE_TOTAL_SIZE (RESTORE_STACK_SIZE + RESTORE_MEM_SIZE)

namespace dmtcp
{
class ProcessInfo
{
  public:
    enum ElfType {
      Elf_32,
      Elf_64
    };

#ifdef JALIB_ALLOCATOR
    static void *operator new(size_t nbytes, void *p) { return p; }

    static void *operator new(size_t nbytes) { JALLOC_HELPER_NEW(nbytes); }

    static void operator delete(void *p) { JALLOC_HELPER_DELETE(p); }
#endif // ifdef JALIB_ALLOCATOR
    ProcessInfo();
    static ProcessInfo &instance();
    void init();
    void postExec();
    void resetOnFork();
    void preCkpt();
    void restart();
    void restoreProcessGroupInfo();
    void restoreHeap();
    void growStack();

    void insertChild(pid_t virtualPid, UniquePid uniquePid);
    void eraseChild(pid_t virtualPid);

    bool beginPthreadJoin(pthread_t thread);
    void endPthreadJoin(pthread_t thread);
    void clearPthreadJoinState(pthread_t thread);

    void refresh();
    void refreshChildTable();
    void setRootOfProcessTree() { _isRootOfProcessTree = true; }

    bool isRootOfProcessTree() const { return _isRootOfProcessTree; }

    void serialize(jalib::JBinarySerializer &o);

    UniquePid compGroup() { return _compGroup; }

    void compGroup(UniquePid cg) { _compGroup = cg; }

    uint32_t numPeers() { return _numPeers; }

    void numPeers(uint32_t np) { _numPeers = np; }

    bool noCoordinator() { return _noCoordinator; }

    void noCoordinator(bool nc) { _noCoordinator = nc; }

    pid_t pid() const { return _pid; }

    pid_t sid() const { return _sid; }

    uint32_t get_generation() { return _generation; }

    void set_generation(uint32_t generation) { _generation = generation; }

    uint32_t numCheckpoints() { return _numCheckpoints; }

    uint32_t numRestarts() { return _numRestarts; }

    uint32_t incrementNumCheckpoints() { return _numCheckpoints++; }

    uint32_t incrementNumRestarts() { return _numRestarts++; }

    void processRlimit();
    void calculateArgvAndEnvSize();
#ifdef RESTORE_ARGV_AFTER_RESTART
    void restoreArgvAfterRestart(char *mtcpRestoreArgvStartAddr);
#endif // ifdef RESTORE_ARGV_AFTER_RESTART
    size_t argvSize() { return _argvSize; }

    size_t envSize() { return _envSize; }

    const string &procname() const { return _procname; }

    const string &procSelfExe() const { return _procSelfExe; }

    const string &hostname() const { return _hostname; }

    const UniquePid &upid() const { return _upid; }

    const UniquePid &uppid() const { return _uppid; }

    bool isOrphan() const { return _ppid == 1; }

    bool isSessionLeader() const { return _pid == _sid; }

    bool isGroupLeader() const { return _pid == _gid; }

    bool isForegroundProcess() const { return _gid == _fgid; }

    bool isChild(const UniquePid &upid);

    int elfType() const { return _elfType; }

    uint64_t savedBrk(void) const { return _savedBrk; }

    uint64_t restoreBufAddr(void) const { return _restoreBufAddr; }

    uint64_t restoreBufLen(void) const { return RESTORE_TOTAL_SIZE; }

    uint64_t vdsoStart(void) const { return _vdsoStart; }

    uint64_t vdsoEnd(void) const { return _vdsoEnd; }

    uint64_t vvarStart(void) const { return _vvarStart; }

    uint64_t vvarEnd(void) const { return _vvarEnd; }

    uint64_t endOfStack(void) const { return _endOfStack; }

    bool vdsoOffsetMismatch(uint64_t f1, uint64_t f2,
                            uint64_t f3, uint64_t f4);

    string getCkptFilename();

    string getCkptFilesSubDir();

    string getCkptDir();

    void setCkptDir(const char *);
    void setCkptFilename(const char *);
    void updateCkptDirFileSubdir(string newCkptDir = "");
    uint32_t getCkptType(void) const { return _ckptType; }
    void setCkptType(int ckpt_type) { _ckptType = ckpt_type; }
    uint32_t getTestMode(void) const { return _testMode; }
    void setTestMode(uint32_t mode) { _testMode = mode; }
    char *getHostName(int rank);
    void setTopology(int num_nodes, char *nameList, int *nodeMap, int *partnerMap);
    int getNumNodes() const { return _topoNumNodes; }
    char *getNameList() const { return _topoNameList; }
    int *getNodeMap() const { return _topoNodeMap; }
    int *getPartnerMap() const { return _topoPartnerMap; }

  private:
    map<pid_t, UniquePid>_childTable;
    map<pthread_t, pthread_t>_pthreadJoinId;
    map<pid_t, pid_t>_sessionIds;
    typedef map<pid_t, UniquePid>::iterator iterator;

    uint32_t _isRootOfProcessTree;
    pid_t _pid;
    pid_t _ppid;
    pid_t _sid;
    pid_t _gid;
    pid_t _fgid;

    // _generation is per-process.  This constrasts with
    // _computation_generation, which is shared among all processes on a host.
    // _computation_generation is updated in shareddata.cpp by:
    // sharedDataHeader->compId._computation_generation = generation;
    // _generation is updated later when this process begins its checkpoint.
    uint32_t _generation;
    uint32_t _numCheckpoints;
    uint32_t _numRestarts;

    uint32_t _numPeers;
    uint32_t _noCoordinator;
    uint32_t _argvSize;
    uint32_t _envSize;
    uint32_t _elfType;

    string _procname;
    string _procSelfExe;
    string _hostname;
    string _launchCWD;
    string _ckptCWD;

    uint32_t _testMode;
    char hostName[HOSTNAME_MAXSIZE];

    //topology information
    int _topoNumNodes;
    char *_topoNameList;
    int *_topoNodeMap;
    int *_topoPartnerMap;
    

    //used for determining which ckpt location to use
    uint32_t _ckptType;

    string _ckptDir[CKPT_GLOBAL+1];
    string _ckptFileName[CKPT_GLOBAL+1];
    string _ckptFilesSubDir[CKPT_GLOBAL+1];

    UniquePid _upid;
    UniquePid _uppid;
    UniquePid _compGroup;

    uint64_t _restoreBufAddr;
    uint64_t _restoreBufLen;

    uint64_t _savedHeapStart;
    uint64_t _savedBrk;

    uint64_t _vdsoStart;
    uint64_t _vdsoEnd;
    uint64_t _vvarStart;
    uint64_t _vvarEnd;
    uint64_t _endOfStack;

    uint64_t _clock_gettime_offset;
    uint64_t _getcpu_offset;
    uint64_t _gettimeofday_offset;
    uint64_t _time_offset;
};
}
#endif /* PROCESS_INFO */

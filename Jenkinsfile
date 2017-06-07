node {
    try {
       stage 'Checkout'
       sh '''
           hostname
       '''
       checkout scm
       sh '''
           curl -X POST -H "X-Auth-User: alecwd" -H "X-Auth-Token: KoPPYriNCbh3vIbg9pF7V6l0z1vU9LmGHpdSpeYFO/+1t+arax/CHwBmo+eT7cygEPBoR59NuOA5u1fcRJwBWVBLSmOeh4ZT/3FMdLKfyrk=" -H "Content-Type: application/json" https://stash.ihme.washington.edu/rest/build-status/1.0/commits/$(git rev-parse HEAD) -d "$(cat <<EOF
{
    "state": "INPROGRESS",
    "key": "CEAM Tests",
    "url": "${BUILD_URL}"
}
EOF
)"
       '''
       stage 'Tests'
       sh '''
           source /ihme/scratch/users/svcceci/.conda/bin/activate /ihme/scratch/users/svcceci/.conda/
           export LD_LIBRARY_PATH=$LD_LIBRARY_PATH://ihme/scratch/users/svcceci/.conda/lib/
           tox --recreate -e py35
       '''

       stage 'Notify'
       currentBuild.result = 'SUCCESS'
       sh '''
           curl -X POST -H "X-Auth-User: alecwd" -H "X-Auth-Token: KoPPYriNCbh3vIbg9pF7V6l0z1vU9LmGHpdSpeYFO/+1t+arax/CHwBmo+eT7cygEPBoR59NuOA5u1fcRJwBWVBLSmOeh4ZT/3FMdLKfyrk=" -H "Content-Type: application/json" https://stash.ihme.washington.edu/rest/build-status/1.0/commits/$(git rev-parse HEAD) -d "$(cat <<EOF
{
    "state": "SUCCESSFUL",
    "key": "CEAM Tests",
    "url": "${BUILD_URL}"
}
EOF
)"

       '''
    } catch (err) {
        currentBuild.result = "FAILURE"
        sh '''
            curl -X POST -H "X-Auth-User: alecwd" -H "X-Auth-Token: KoPPYriNCbh3vIbg9pF7V6l0z1vU9LmGHpdSpeYFO/+1t+arax/CHwBmo+eT7cygEPBoR59NuOA5u1fcRJwBWVBLSmOeh4ZT/3FMdLKfyrk=" -H "Content-Type: application/json" https://stash.ihme.washington.edu/rest/build-status/1.0/commits/$(git rev-parse HEAD) -d "$(cat <<EOF
{
    "state": "FAILED",
    "key": "CEAM Tests",
    "url": "${BUILD_URL}"
}
EOF
)"

            /usr/sbin/sendmail "$(git --no-pager show -s --format='%an <%ae>' HEAD)" <<EOF
Subject: Build failure on branch: ${BRANCH_NAME}

Your commit ($(git rev-parse HEAD)) seems to have broken the build. Take a look: ${BUILD_URL}/
EOF
        '''
    }
}

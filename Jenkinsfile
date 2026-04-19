// =============================================================================
// movie-finder-rag — Jenkins declarative pipeline
//
// Pipeline modes (Jenkins Multibranch Pipeline):
//   PR build       — every pull request: Lint + Typecheck + Test (no image build)
//   Main build     — push to main: Lint + Typecheck + Test (no image build)
//   Manual ops     — "Build with Parameters" with RUN_INGESTION/RUN_BACKUP=true:
//                    Lint + Typecheck + Test + optional ingest/backup against live vector store
//
// NOTE: This image is NOT pushed to ACR. The rag pipeline is an offline
// one-shot ingestion tool run manually. Only backend and frontend images are
// published to ACR.
//
// Required Jenkins Credential IDs (see docs/devops-setup.md):
//   qdrant-url           (Secret Text)   — Manual ingest / backup only
//   qdrant-api-key-rw    (Secret Text)   — Manual ingest / backup only
//   openai-api-key       (Secret Text)   — Manual ingest only
//   kaggle-api-token     (Secret Text)   — Manual ingest only
//
// Required Jenkins plugins: Docker Pipeline, JUnit, Coverage, Credentials Binding
// =============================================================================

pipeline {
    agent any

    options {
        buildDiscarder(logRotator(numToKeepStr: '20'))
        timeout(time: 45, unit: 'MINUTES')
        disableConcurrentBuilds(abortPrevious: true)
    }

    parameters {
        booleanParam(
            name: 'RUN_INGESTION',
            defaultValue: false,
            description: 'Run the offline ingestion pipeline against Qdrant Cloud (requires live credentials).'
        )
        booleanParam(
            name: 'RUN_BACKUP',
            defaultValue: false,
            description: 'Run the backup utility against a live vector store and archive the local artifact.'
        )
        string(
            name: 'COLLECTION_NAME',
            defaultValue: 'movies',
            description: 'Collection name to ingest and/or back up. Defaults to "movies".'
        )
        choice(
            name: 'VECTOR_STORE',
            choices: ['qdrant'],
            description: 'Vector store type for the backup utility. Extend as new adapters are added.'
        )
    }

    environment {
        SERVICE_NAME = 'movie-finder-rag'
        DOCKER_BUILDKIT = '1'
        // Isolate compose project per build so parallel CI runs don't collide.
        COMPOSE_PROJECT_NAME = "movie-finder-rag-ci-${env.BUILD_NUMBER}"
    }

    stages {

        // ------------------------------------------------------------------ //
        stage('Initialize') {
            steps {
                sh 'make init'
            }
        }

        // ------------------------------------------------------------------ //
        stage('Lint + Typecheck') {
            parallel {
                stage('Lint') {
                    steps { sh 'make lint' }
                }
                stage('Typecheck') {
                    steps { sh 'make typecheck' }
                }
            }
        }

        // ------------------------------------------------------------------ //
        stage('Test') {
            steps {
                sh 'make test-coverage'
            }
            post {
                always {
                    junit allowEmptyResults: true, testResults: 'reports/junit.xml'
                    recordCoverage(
                        tools: [
                            [parser: 'COBERTURA', pattern: 'reports/coverage.xml']
                        ],
                        id: 'coverage',
                        name: 'RAG Coverage',
                        sourceCodeRetention: 'EVERY_BUILD',
                        sourceDirectories: [[path: 'src']],
                        failOnError: false,
                        qualityGates: [
                            [threshold: 80.0, metric: 'LINE', baseline: 'PROJECT'],
                            [threshold: 80.0, metric: 'BRANCH', baseline: 'PROJECT']
                        ]
                    )
                }
            }
        }

        // ------------------------------------------------------------------ //
        stage('Ingest') {
            when {
                expression { params.RUN_INGESTION == true }
            }
            environment {
                QDRANT_URL        = credentials('qdrant-url')
                QDRANT_API_KEY_RW = credentials('qdrant-api-key-rw')
                OPENAI_API_KEY    = credentials('openai-api-key')
                KAGGLE_API_TOKEN  = credentials('kaggle-api-token')
            }
            steps {
                script {
                    // env blocks only allow literals or credentials() calls.
                    // Use script {} to evaluate the default-value expression.
                    env.QDRANT_COLLECTION_NAME = params.COLLECTION_NAME ?: 'movies'
                    echo "Target collection: ${env.QDRANT_COLLECTION_NAME}"
                    sh 'make ingest'
                }
            }
            post {
                always {
                    archiveArtifacts artifacts: 'ingestion-outputs.env', allowEmptyArchive: true
                }
            }
        }

        // ------------------------------------------------------------------ //
        stage('Backup') {
            when {
                expression { params.RUN_BACKUP == true || params.RUN_INGESTION == true }
            }
            environment {
                QDRANT_URL        = credentials('qdrant-url')
                QDRANT_API_KEY_RW = credentials('qdrant-api-key-rw')
            }
            steps {
                script {
                    env.VECTOR_STORE = params.VECTOR_STORE ?: 'qdrant'
                    env.BACKUP_COLLECTION_NAME = params.COLLECTION_NAME ?: 'movies'
                    env.BACKUP_OUTPUT_ROOT = 'outputs/backups/chromadb'
                    env.VECTOR_STORE_URL = env.QDRANT_URL
                    env.VECTOR_STORE_API_KEY = env.QDRANT_API_KEY_RW
                    echo "Backup source: ${env.VECTOR_STORE}"
                    echo "Backup collection: ${env.BACKUP_COLLECTION_NAME}"
                    sh 'make backup'
                }
            }
            post {
                always {
                    archiveArtifacts artifacts: 'outputs/backups/**', allowEmptyArchive: true
                }
            }
        }

    }

    post {
        always {
            sh 'make clean || true'
            sh 'make ci-down || true'
            cleanWs()
        }
        failure {
            echo "Pipeline failed on ${env.BRANCH_NAME ?: 'manual trigger'} — check logs above."
        }
        success {
            script {
                if (params.RUN_INGESTION) {
                    echo "Ingestion into '${env.QDRANT_COLLECTION_NAME}' completed successfully."
                }
                if (params.RUN_BACKUP || params.RUN_INGESTION) {
                    echo "Backup artifact archived from outputs/backups/."
                }
            }
        }
    }
}

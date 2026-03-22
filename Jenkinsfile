// =============================================================================
// movie-finder-rag — Jenkins declarative pipeline
//
// Triggers:
//   • PR validation  — every pull request to main (lint + test only)
//   • Release        — every git tag matching v* (lint + test + build image)
//   • Manual ingest  — Build with Parameters → check RUN_INGESTION
//                      (runs the full dataset ingestion job)
//
// Required Jenkins credentials:
//   docker-registry-url  — Docker registry base URL
//   qdrant-endpoint      — Qdrant Cloud endpoint URL   (for manual ingest)
//   qdrant-api-key       — Qdrant Cloud API key        (for manual ingest)
//   openai-api-key       — OpenAI API key              (for manual ingest)
//   kaggle-username      — Kaggle account username     (for manual ingest)
//   kaggle-key           — Kaggle API key              (for manual ingest)
//
// Required Jenkins plugins:
//   Docker Pipeline, JUnit, Cobertura, Credentials Binding
// =============================================================================

pipeline {
    agent none

    options {
        buildDiscarder(logRotator(numToKeepStr: '20'))
        timeout(time: 60, unit: 'MINUTES')   // Ingestion can take >30 min
        disableConcurrentBuilds()            // Never run two ingestions in parallel
    }

    parameters {
        booleanParam(
            name: 'RUN_INGESTION',
            defaultValue: false,
            description: 'Trigger a full dataset ingestion run (manual). Requires all ingestion credentials.'
        )
        choice(
            name: 'VECTOR_STORE',
            choices: ['qdrant', 'chromadb'],
            description: 'Target vector store backend for this ingestion run.'
        )
        string(
            name: 'COLLECTION_NAME',
            defaultValue: 'movies',
            description: 'Qdrant collection name to write to. Use a new name for a fresh ingest.'
        )
        booleanParam(
            name: 'FORCE_DOWNLOAD',
            defaultValue: false,
            description: 'Force re-download of the Kaggle dataset even if cached.'
        )
    }

    environment {
        SERVICE_NAME = 'movie-finder-rag'
        UV_IMAGE     = 'ghcr.io/astral-sh/uv:python3.13-bookworm-slim'
        DOCKER_IMAGE = 'docker:24-dind'
    }

    stages {

        // ------------------------------------------------------------------ //
        stage('Lint') {
            agent {
                docker {
                    image "${UV_IMAGE}"
                }
            }
            steps {
                sh 'uv sync --frozen --group lint'
                sh 'uv run ruff check src/ tests/'
                sh 'uv run ruff format --check src/ tests/'
                sh 'uv run mypy src/'
            }
        }

        // ------------------------------------------------------------------ //
        stage('Test') {
            agent {
                docker {
                    image "${UV_IMAGE}"
                }
            }
            steps {
                sh 'uv sync --frozen --group test'
                sh '''
                    uv run pytest tests/ \
                        --cov=src \
                        --cov-report=xml:coverage.xml \
                        --junitxml=test-results.xml \
                        -v --tb=short
                '''
            }
            post {
                always {
                    junit allowEmptyResults: true, testResults: 'test-results.xml'
                    cobertura coberturaReportFile: 'coverage.xml',
                              onlyStable: false,
                              failNoReports: false
                }
            }
        }

        // ------------------------------------------------------------------ //
        stage('Build Image') {
            when {
                anyOf {
                    branch 'main'
                    buildingTag()
                    expression { params.RUN_INGESTION }
                }
            }
            agent {
                docker {
                    image "${DOCKER_IMAGE}"
                    args '--privileged -v /var/run/docker.sock:/var/run/docker.sock'
                }
            }
            environment {
                DOCKER_REGISTRY = credentials('docker-registry-url')
            }
            steps {
                script {
                    def tag = env.GIT_TAG_NAME ?: env.GIT_COMMIT.take(8)
                    env.IMAGE_TAG = "${DOCKER_REGISTRY}/${SERVICE_NAME}:${tag}"
                }
                sh "docker build -t ${IMAGE_TAG} ."
                script {
                    if (env.BRANCH_NAME == 'main' || buildingTag()) {
                        sh "docker push ${IMAGE_TAG}"
                        if (env.BRANCH_NAME == 'main') {
                            sh "docker tag ${IMAGE_TAG} ${DOCKER_REGISTRY}/${SERVICE_NAME}:latest"
                            sh "docker push ${DOCKER_REGISTRY}/${SERVICE_NAME}:latest"
                        }
                    }
                }
            }
        }

        // ------------------------------------------------------------------ //
        stage('Ingest') {
            // Manual trigger only — requires RUN_INGESTION=true parameter
            when {
                expression { params.RUN_INGESTION == true }
            }
            agent {
                docker {
                    image "${DOCKER_IMAGE}"
                    args '--privileged -v /var/run/docker.sock:/var/run/docker.sock'
                }
            }
            environment {
                QDRANT_ENDPOINT   = credentials('qdrant-endpoint')
                QDRANT_API_KEY    = credentials('qdrant-api-key')
                OPENAI_API_KEY    = credentials('openai-api-key')
                KAGGLE_USERNAME   = credentials('kaggle-username')
                KAGGLE_KEY        = credentials('kaggle-key')
                QDRANT_COLLECTION = "${params.COLLECTION_NAME}"
                VECTOR_STORE      = "${params.VECTOR_STORE}"
            }
            steps {
                echo "Starting ingestion run into collection '${params.COLLECTION_NAME}' on ${params.VECTOR_STORE}"
                sh """
                    docker run --rm \
                        -e QDRANT_ENDPOINT \
                        -e QDRANT_API_KEY \
                        -e QDRANT_COLLECTION \
                        -e OPENAI_API_KEY \
                        -e KAGGLE_USERNAME \
                        -e KAGGLE_KEY \
                        -e VECTOR_STORE \
                        -e FORCE_DOWNLOAD=${params.FORCE_DOWNLOAD} \
                        ${IMAGE_TAG}
                """
            }
            post {
                success {
                    // Archive outputs for the chain team to consume.
                    sh """
                        printf '# rag_ingestion outputs — share with chain team\\n' > ingestion-outputs.env
                        printf '# Build: ${env.BUILD_NUMBER}\\n' >> ingestion-outputs.env
                        printf 'QDRANT_COLLECTION=${params.COLLECTION_NAME}\\n' >> ingestion-outputs.env
                        printf 'EMBEDDING_MODEL=text-embedding-3-large\\n' >> ingestion-outputs.env  # pragma: allowlist secret
                        printf 'EMBEDDING_DIMENSION=3072\\n' >> ingestion-outputs.env
                    """
                    archiveArtifacts artifacts: 'ingestion-outputs.env', fingerprint: true
                    echo """
=======================================================================
  INGESTION COMPLETE
  Collection : ${params.COLLECTION_NAME}
  Model      : text-embedding-3-large

  ACTION REQUIRED:
  Download 'ingestion-outputs.env' artifact and update the chain
  team's Jenkins 'qdrant-collection' credential with the new value.
=======================================================================
                    """
                }
            }
        }

    }

    post {
        always {
            cleanWs()
        }
        failure {
            echo "Pipeline failed on branch ${env.BRANCH_NAME} — check logs above."
        }
    }
}

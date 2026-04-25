// =============================================================================
// movie-finder-rag — Jenkins declarative pipeline
//
// Pipeline modes (Jenkins Multibranch Pipeline):
//   PR build       — every pull request: Lint + Typecheck + Test
//   Main build     — push to main: Lint + Typecheck + Test
//   Manual ops     — "Build with Parameters" with RUN_INGESTION/RUN_BACKUP=true:
//                    Lint + Typecheck + Test + optional ingest + validate + backup
//
// Repo-side guards also skip draft PRs and non-main branch jobs. The Jenkins
// multibranch source should still be configured to discover only main + PRs;
// these guards prevent expensive work if Jenkins indexes extra branches.
//
// NOTE: This image is NOT pushed to ACR. The rag pipeline is an offline
// one-shot ingestion tool run manually. Only backend and frontend images are
// published to ACR.
//
// Docker-only contract:
// - All commands run through the repo Makefile and Docker Compose.
// - Do not assume host Python, host Ollama, or localhost-only services.
// - ChromaDB may be local to the mounted workspace path; remote backends remain external.
//
// ADR 0008 runtime contract:
// - Embedding provider/model are selected via EMBEDDING_PROVIDER / EMBEDDING_MODEL.
// - Final target name is derived from COLLECTION_PREFIX + sanitized model + dimension.
// - Backups are normalized into a portable ChromaDB artifact regardless of source backend.
//
// Required Jenkins Credential IDs (see docs/devops-setup.md):
//   qdrant-url           (Secret Text)   — qdrant ingest / validate / backup
//   qdrant-api-key-rw    (Secret Text)   — qdrant ingest / validate / backup
//   openai-api-key       (Secret Text)   — openai embeddings
//   google-api-key       (Secret Text)   — google embeddings
//   ollama-api-key       (Secret Text)   — ollama cloud, when used
//   pinecone-api-key     (Secret Text)   — pinecone backend, when used
//   pgvector-dsn         (Secret Text)   — pgvector backend, when used
//   kaggle-api-token     (Secret Text)   — dataset download
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
            description: 'Run the offline ingestion pipeline against the configured vector store.'
        )
        booleanParam(
            name: 'RUN_BACKUP',
            defaultValue: false,
            description: 'Run the backup utility and archive the generated artifact.'
        )
        choice(
            name: 'EMBEDDING_PROVIDER',
            choices: ['openai', 'ollama', 'huggingface', 'sentence-transformers', 'google'],
            description: 'ADR 0008 embedding provider selection.'
        )
        string(
            name: 'EMBEDDING_MODEL',
            defaultValue: 'text-embedding-3-large',
            description: 'Embedding model identifier for the selected provider.'
        )
        string(
            name: 'EMBEDDING_DIMENSIONS',
            defaultValue: '',
            description: 'Optional output dimension override for providers that support it.'
        )
        password(
            name: 'EMBEDDING_API_KEY',
            defaultValue: '',
            description: 'Optional API key override for OpenAI, Google, or Ollama cloud.'
        )
        string(
            name: 'COLLECTION_PREFIX',
            defaultValue: 'movies',
            description: 'Qdrant collection prefix. Final target is resolved dynamically per ADR 0008.'
        )
        choice(
            name: 'VECTOR_STORE',
            choices: ['qdrant', 'chromadb', 'pinecone', 'pgvector'],
            description: 'Vector store backend for ingestion and validation.'
        )
        string(
            name: 'VECTOR_STORE_URL',
            defaultValue: '',
            description: 'Optional qdrant URL override for qdrant backups.'
        )
        password(
            name: 'VECTOR_STORE_API_KEY',
            defaultValue: '',
            description: 'Optional qdrant or pinecone API key override.'
        )
        string(
            name: 'OLLAMA_URL',
            defaultValue: 'http://ollama:11434',
            description: 'Docker-reachable Ollama base URL when EMBEDDING_PROVIDER=ollama.'
        )
        string(
            name: 'CHROMADB_PERSIST_PATH',
            defaultValue: 'outputs/chromadb/local',
            description: 'Persistent path for chromadb ingestion and backup.'
        )
        string(
            name: 'PINECONE_INDEX_NAME',
            defaultValue: 'movie-finder-rag',
            description: 'Pinecone index name when VECTOR_STORE=pinecone.'
        )
        string(
            name: 'PINECONE_INDEX_HOST',
            defaultValue: '',
            description: 'Optional Pinecone host override.'
        )
        string(
            name: 'PINECONE_CLOUD',
            defaultValue: 'aws',
            description: 'Pinecone serverless cloud when an index must be created.'
        )
        string(
            name: 'PINECONE_REGION',
            defaultValue: 'us-east-1',
            description: 'Pinecone serverless region when an index must be created.'
        )
        password(
            name: 'PGVECTOR_DSN_OVERRIDE',
            defaultValue: '',
            description: 'Optional pgvector PostgreSQL DSN override.'
        )
        string(
            name: 'PGVECTOR_SCHEMA',
            defaultValue: 'public',
            description: 'Schema token used when VECTOR_STORE=pgvector.'
        )
        string(
            name: 'WITH_PROVIDERS',
            defaultValue: '',
            description: 'Docker build extra groups, for example `local`.'
        )
        string(
            name: 'VALIDATION_QUERY',
            defaultValue: 'A time-travel movie with a scientist and a DeLorean',
            description: 'Smoke-test query for the post-ingest validation stage.'
        )
    }

    environment {
        SERVICE_NAME = 'movie-finder-rag'
        DOCKER_BUILDKIT = '1'
        COMPOSE_PROJECT_NAME = "movie-finder-rag-ci-${env.BUILD_NUMBER}"
    }

    stages {
        // ------------------------------------------------------------------ //
        stage('CI Eligibility') {
            steps {
                script {
                    if (shouldRunPipeline()) {
                        echo "RAG CI enabled for ${env.CHANGE_ID ? "PR-${env.CHANGE_ID}" : env.BRANCH_NAME}."
                    } else {
                        echo "RAG CI skipped for ${env.CHANGE_ID ? "draft PR-${env.CHANGE_ID}" : "branch ${env.BRANCH_NAME}"}."
                    }
                }
            }
        }

        // ------------------------------------------------------------------ //
        stage('Initialize') {
            when {
                expression { shouldRunPipeline() }
            }
            steps {
                script {
                    env.WITH_PROVIDERS = params.WITH_PROVIDERS ?: ''
                    sh 'make init'
                }
            }
        }

        // ------------------------------------------------------------------ //
        stage('Lint + Typecheck') {
            when {
                expression { shouldRunPipeline() }
            }
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
            when {
                expression { shouldRunPipeline() }
            }
            steps {
                sh 'make test-coverage'
            }
            post {
                always {
                    junit allowEmptyResults: true, testResults: 'reports/junit.xml'
                    recordCoverage(
                        tools: [[parser: 'COBERTURA', pattern: 'reports/coverage.xml']],
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
                expression { shouldRunPipeline() && params.RUN_INGESTION == true }
            }
            environment {
                QDRANT_URL        = credentials('qdrant-url')
                QDRANT_API_KEY_RW = credentials('qdrant-api-key-rw')
                OPENAI_API_KEY    = credentials('openai-api-key')
                GOOGLE_API_KEY    = credentials('google-api-key')
                OLLAMA_API_KEY    = credentials('ollama-api-key')
                PINECONE_API_KEY  = credentials('pinecone-api-key')
                PGVECTOR_DSN      = credentials('pgvector-dsn')
                KAGGLE_API_TOKEN  = credentials('kaggle-api-token')
            }
            steps {
                script {
                    configureRuntimeEnv()
                    sh 'make ingest'
                }
            }
        }

        // ------------------------------------------------------------------ //
        stage('Post-Ingest Validate') {
            when {
                expression { shouldRunPipeline() && params.RUN_INGESTION == true }
            }
            environment {
                QDRANT_URL        = credentials('qdrant-url')
                QDRANT_API_KEY_RW = credentials('qdrant-api-key-rw')
                OPENAI_API_KEY    = credentials('openai-api-key')
                GOOGLE_API_KEY    = credentials('google-api-key')
                OLLAMA_API_KEY    = credentials('ollama-api-key')
                PINECONE_API_KEY  = credentials('pinecone-api-key')
                PGVECTOR_DSN      = credentials('pgvector-dsn')
            }
            steps {
                script {
                    configureRuntimeEnv()
                    sh 'make validate'
                }
            }
        }

        // ------------------------------------------------------------------ //
        stage('Backup') {
            when {
                expression {
                    shouldRunPipeline() && (params.RUN_BACKUP == true || params.RUN_INGESTION == true)
                }
            }
            environment {
                QDRANT_URL        = credentials('qdrant-url')
                QDRANT_API_KEY_RW = credentials('qdrant-api-key-rw')
                OPENAI_API_KEY    = credentials('openai-api-key')
                GOOGLE_API_KEY    = credentials('google-api-key')
                OLLAMA_API_KEY    = credentials('ollama-api-key')
                PINECONE_API_KEY  = credentials('pinecone-api-key')
                PGVECTOR_DSN      = credentials('pgvector-dsn')
            }
            steps {
                script {
                    configureRuntimeEnv()
                    sh 'make backup'
                }
            }
        }

        // ------------------------------------------------------------------ //
        stage('Archive Artifacts') {
            when {
                expression {
                    shouldRunPipeline() && (params.RUN_INGESTION == true || params.RUN_BACKUP == true)
                }
            }
            steps {
                archiveArtifacts(
                    artifacts: 'ingestion-outputs.env,outputs/reports/**,outputs/backups/**',
                    allowEmptyArchive: true
                )
            }
        }
    }

    post {
        always {
            script {
                if (shouldRunPipeline()) {
                    sh 'make clean || true'
                    sh 'make ci-down || true'
                } else {
                    echo 'Skipping Docker cleanup because no RAG CI stages ran.'
                }
                cleanWs()
            }
        }
    }
}

// -------------------------------------------------------------------------- //
def shouldRunPipeline() {
    if (env.RAG_CI_ELIGIBLE?.trim()) {
        return env.RAG_CI_ELIGIBLE == 'true'
    }

    def eligible = resolvePipelineEligibility()
    env.RAG_CI_ELIGIBLE = eligible.toString()
    return eligible
}

// -------------------------------------------------------------------------- //
def resolvePipelineEligibility() {
    if (env.CHANGE_ID) {
        return !isDraftPullRequest()
    }

    return env.BRANCH_NAME == 'main'
}

// -------------------------------------------------------------------------- //
def isDraftPullRequest() {
    withCredentials([
        usernamePassword(
            credentialsId: 'github-pat',
            usernameVariable: 'GITHUB_USER',
            passwordVariable: 'GITHUB_TOKEN'
        )
    ]) {
        withEnv(["PR_NUMBER=${env.CHANGE_ID}"]) {
            def draft = sh(
                script: '''
                    set +x
                    python3 - <<'PY'
import json
import os
import urllib.request

request = urllib.request.Request(
    f"https://api.github.com/repos/aharbii/movie-finder-rag/pulls/{os.environ['PR_NUMBER']}",
    headers={
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {os.environ['GITHUB_TOKEN']}",
    },
)
with urllib.request.urlopen(request, timeout=20) as response:
    print(str(bool(json.load(response).get("draft"))).lower())
PY
                ''',
                returnStdout: true,
            ).trim()
            return draft == 'true'
        }
    }
}

// -------------------------------------------------------------------------- //
def configureRuntimeEnv() {
    env.WITH_PROVIDERS = params.WITH_PROVIDERS ?: ''
    env.EMBEDDING_PROVIDER = params.EMBEDDING_PROVIDER ?: 'openai'
    env.EMBEDDING_MODEL = params.EMBEDDING_MODEL ?: 'text-embedding-3-large'
    env.EMBEDDING_DIMENSIONS = params.EMBEDDING_DIMENSIONS ?: ''
    env.OLLAMA_URL = params.OLLAMA_URL ?: env.OLLAMA_URL
    env.QDRANT_COLLECTION_PREFIX = params.COLLECTION_PREFIX ?: 'movies'
    env.VECTOR_STORE = params.VECTOR_STORE ?: 'qdrant'
    env.VALIDATION_QUERY = params.VALIDATION_QUERY ?: 'A time-travel movie with a scientist and a DeLorean'
    env.CHROMADB_PERSIST_PATH = params.CHROMADB_PERSIST_PATH ?: env.CHROMADB_PERSIST_PATH ?: 'outputs/chromadb/local'
    env.PINECONE_INDEX_NAME = params.PINECONE_INDEX_NAME ?: env.PINECONE_INDEX_NAME ?: 'movie-finder-rag'
    env.PINECONE_INDEX_HOST = params.PINECONE_INDEX_HOST ?: env.PINECONE_INDEX_HOST
    env.PINECONE_CLOUD = params.PINECONE_CLOUD ?: env.PINECONE_CLOUD ?: 'aws'
    env.PINECONE_REGION = params.PINECONE_REGION ?: env.PINECONE_REGION ?: 'us-east-1'
    env.PGVECTOR_SCHEMA = params.PGVECTOR_SCHEMA ?: env.PGVECTOR_SCHEMA ?: 'public'

    if (params.EMBEDDING_API_KEY?.trim()) {
        if (env.EMBEDDING_PROVIDER == 'openai') {
            env.OPENAI_API_KEY = params.EMBEDDING_API_KEY
        } else if (env.EMBEDDING_PROVIDER == 'google') {
            env.GOOGLE_API_KEY = params.EMBEDDING_API_KEY
        } else if (env.EMBEDDING_PROVIDER == 'ollama') {
            env.OLLAMA_API_KEY = params.EMBEDDING_API_KEY
        }
    }

    if (env.VECTOR_STORE == 'qdrant') {
        env.VECTOR_STORE_URL = params.VECTOR_STORE_URL ?: env.QDRANT_URL
        env.VECTOR_STORE_API_KEY = params.VECTOR_STORE_API_KEY?.trim() ? params.VECTOR_STORE_API_KEY : env.QDRANT_API_KEY_RW
    } else if (env.VECTOR_STORE == 'pinecone') {
        env.PINECONE_INDEX_HOST = params.PINECONE_INDEX_HOST ?: params.VECTOR_STORE_URL ?: env.PINECONE_INDEX_HOST
        env.PINECONE_API_KEY = params.VECTOR_STORE_API_KEY?.trim() ? params.VECTOR_STORE_API_KEY : env.PINECONE_API_KEY
        env.VECTOR_STORE_URL = env.PINECONE_INDEX_HOST
        env.VECTOR_STORE_API_KEY = env.PINECONE_API_KEY
    } else if (env.VECTOR_STORE == 'pgvector') {
        env.PGVECTOR_DSN = params.PGVECTOR_DSN_OVERRIDE?.trim() ? params.PGVECTOR_DSN_OVERRIDE : env.PGVECTOR_DSN
        env.VECTOR_STORE_URL = env.PGVECTOR_DSN
        env.VECTOR_STORE_API_KEY = ''
    } else if (env.VECTOR_STORE == 'chromadb') {
        env.VECTOR_STORE_URL = ''
        env.VECTOR_STORE_API_KEY = ''
    }
}

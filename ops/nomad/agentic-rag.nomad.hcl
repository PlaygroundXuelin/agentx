variable "environment" {
  type    = string
  default = "dev"
}

variable "namespace" {
  type    = string
  default = "default"
}

variable "datacenters" {
  type    = list(string)
  default = ["dc1"]
}

variable "catalog_image" {
  type    = string
  default = "agentic-rag/catalog:dev"
}

variable "datasets_image" {
  type    = string
  default = "agentic-rag/datasets:dev"
}

variable "catalog_count" {
  type    = number
  default = 1
}

variable "datasets_count" {
  type    = number
  default = 1
}

variable "catalog_config_path" {
  type    = string
  default = "/app/catalog/deploy/configs/nomad.yaml"
}

variable "datasets_config_path" {
  type    = string
  default = "/app/datasets/deploy/configs/nomad.yaml"
}

job "agentic-rag" {
  datacenters = var.datacenters
  namespace   = var.namespace
  type        = "service"

  meta {
    environment = var.environment
  }

  group "catalog" {
    count = var.catalog_count

    network {
      mode = "bridge"
      port "http" {
        to     = 9000
        static = 9000
      }
    }

    service {
      name     = "catalog-${var.environment}"
      port     = "http"
      provider = "consul"

      # Register against the allocation IP so health checks run inside the bridge network.
      address_mode = "alloc"

      connect {
        sidecar_service {}
        sidecar_task {
          resources {
            cpu    = 100
            memory = 64
          }
        }
      }

      check {
        name     = "catalog-http"
        type     = "http"
        method   = "GET"
        path     = "/v1/ping"   # or "/health"
        port     = "http"
        address_mode = "alloc"
        interval = "10s"
        timeout  = "5s"         # a bit more lenient at startup
      }
    }

    task "catalog" {
      driver = "docker"

      config {
        image = var.catalog_image
        ports = ["http"]
      }

      env {
        APP_CONFIG_PATH = var.catalog_config_path
      }

      resources {
        cpu    = 500
        memory = 512
      }

      restart {
        attempts = 10
        interval = "5m"
        delay    = "15s"
        mode     = "delay"
      }
    }
  }

  group "datasets" {
    count = var.datasets_count

    network {
      mode = "bridge"
      port "http" {
        to     = 8100
        static = 8100
      }
    }

    service {
      name     = "datasets-${var.environment}"
      port     = "http"
      provider = "consul"

      address_mode = "alloc"

      connect {
        sidecar_service {
          proxy {
            upstreams {
              destination_name = "catalog-${var.environment}"
              destination_namespace = var.namespace
              local_bind_address = "0.0.0.0"
              local_bind_port  = 9191
            }
          }
        }
        sidecar_task {
          resources {
            cpu    = 100
            memory = 64
          }
        }
      }

      check {
        name     = "datasets-http"
        type     = "http"
        method   = "GET"
        path     = "/v1/ping"   # or "/health"
        port     = "http"
        address_mode = "alloc"
        interval = "10s"
        timeout  = "5s"
      }
    }

    task "datasets" {
      driver = "docker"

      config {
        image = var.datasets_image
        ports = ["http"]
      }

      env {
        APP_CONFIG_PATH = var.datasets_config_path
      }

      template {
        data = <<-EOT
SVC_CATALOG_URL=http://127.0.0.1:9191
FS__ROOT={{ env "NOMAD_ALLOC_DIR" }}/data/datasets
EOT
        destination = "secrets/upstreams.env"
        env         = true
      }

      resources {
        cpu    = 500
        memory = 512
      }

      restart {
        attempts = 10
        interval = "5m"
        delay    = "15s"
        mode     = "delay"
      }
    }
  }
}

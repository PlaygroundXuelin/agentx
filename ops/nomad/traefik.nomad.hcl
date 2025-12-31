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

variable "traefik_image" {
  type    = string
  default = "traefik:v3.1"
}

variable "traefik_count" {
  type    = number
  default = 1
}

variable "traefik_nomad_endpoint" {
  type    = string
  default = "http://10.0.0.166:4646"
}

job "traefik" {
  datacenters = var.datacenters
  namespace   = var.namespace
  type        = "service"

  meta {
    environment = var.environment
  }

  group "traefik" {
    count = var.traefik_count

    network {
      mode = "host"

      port "http" {
        to     = 8080
        static = 8080
      }

      port "dashboard" {
        to     = 8081
        static = 8081
      }
    }

    service {
      name     = "traefik-${var.environment}"
      port     = "http"
      provider = "nomad"
      tags     = ["traefik.enable=false"]

      check {
        name     = "traefik-http"
        type     = "http"
        port     = "dashboard"
        path     = "/ping"
        interval = "10s"
        timeout  = "2s"
      }
    }

    task "traefik" {
      driver = "docker"

      config {
        image = var.traefik_image
        network_mode = "host"
        ports = ["http", "dashboard"]
        args = [
          "--api.dashboard=true",
          "--api.insecure=true",
          "--entryPoints.web.address=:8080",
          "--entryPoints.traefik.address=:8081",
          "--ping=true",
          "--ping.entryPoint=traefik",
          "--providers.nomad=true",
          "--providers.nomad.exposedByDefault=false",
          "--providers.nomad.endpoint.address=${var.traefik_nomad_endpoint}"
        ]
      }

      resources {
        cpu    = 300
        memory = 256
      }

      restart {
        attempts = 5
        interval = "10m"
        delay    = "15s"
        mode     = "delay"
      }
    }
  }
}

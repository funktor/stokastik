locals {
  region = "us-east-1"
  tags = {
    Owner       = "amondal"
    Environment = "dev"
  }
}

provider "aws" {
  region = local.region
}

resource "aws_dynamodb_table" "dynamodb" {
  name           = "Profiles"
  billing_mode   = "PROVISIONED"
  read_capacity  = 5
  write_capacity = 5
  hash_key       = "profile_id"

  attribute {
    name = "profile_id"
    type = "S"
  }
  
  stream_enabled = true
  stream_view_type = "NEW_IMAGE"

  tags = local.tags
}

module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "~> 2"

  name = "postgres-abhijit-vpc"
  cidr = "10.99.0.0/18"

  azs              = ["${local.region}a", "${local.region}b", "${local.region}c"]
  public_subnets   = ["10.99.0.0/24", "10.99.1.0/24", "10.99.2.0/24"]
  private_subnets  = ["10.99.3.0/24", "10.99.4.0/24", "10.99.5.0/24"]
  database_subnets = ["10.99.7.0/24", "10.99.8.0/24", "10.99.9.0/24"]

  create_database_subnet_group = true

  tags = local.tags
}

module "security_group" {
  source      = "terraform-aws-modules/security-group/aws"
  version     = "~> 4"
  name        = "postgres-abhijit-sg"
  description = "RDS PostgreSQL security group"
  vpc_id      = module.vpc.vpc_id

  ingress_with_cidr_blocks = [
    {
      from_port   = 5432
      to_port     = 5432
      protocol    = "tcp"
      description = "PostgreSQL access from within VPC"
      cidr_blocks = module.vpc.vpc_cidr_block
    },
  ]

  egress_with_cidr_blocks = [
    {
      from_port   = 5432
      to_port     = 5432
      protocol    = "tcp"
      description = "PostgreSQL access from within VPC"
      cidr_blocks = module.vpc.vpc_cidr_block
    },
  ]

  tags = local.tags
}

resource "aws_db_parameter_group" "parameter_group" {
  name   = "postgres-abhijit-db-pg"
  family = "postgres13"

  tags = local.tags
}

resource "aws_db_instance" "db" {
  identifier                          = "postgres-abhijit-v2"
  engine                              = "postgres"
  engine_version                      = "13.1"
  instance_class                      = "db.t3.small"
  allocated_storage                   = 20
  max_allocated_storage               = 100
  storage_encrypted                   = false
  username                            = "postgres"
  password                            = var.db_password
  port                                = 5432
  multi_az                            = true
  vpc_security_group_ids              = [module.security_group.security_group_id]
  db_subnet_group_name                = module.vpc.database_subnet_group_name
  enabled_cloudwatch_logs_exports     = ["postgresql"]
  parameter_group_name                = aws_db_parameter_group.parameter_group.name
  backup_retention_period             = 7
  skip_final_snapshot                 = true
  deletion_protection                 = false
  performance_insights_enabled        = true
  tags                                = local.tags
}

resource "aws_ecr_repository" "lambda_repository" {
  name                 = "lambda-rds"
  image_tag_mutability = "MUTABLE"
}

resource "aws_lambda_function" "lambda_function" {
  function_name = "lambda-rds"

  role = aws_iam_role.lambda_role.arn

  image_uri    = "${aws_ecr_repository.lambda_repository.repository_url}:latest"
  package_type = "Image"

  vpc_config {
    subnet_ids         = module.vpc.private_subnets
    security_group_ids = [module.security_group.security_group_id]
  }

  timeout = 500
}

resource "aws_iam_role" "lambda_role" {
  name = "lambda-rds-role"

  assume_role_policy = <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "",
      "Effect": "Allow",
      "Principal": {
        "Service": "lambda.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
EOF
}

resource "aws_iam_role_policy_attachment" "lambda_policy_attachement" {
  role       = aws_iam_role.lambda_role.name
  policy_arn = aws_iam_policy.lambda_policy.arn
}

resource "aws_iam_policy" "lambda_policy" {
  name = "lambda-rds-policy"

  policy = <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "logs:CreateLogGroup",
        "logs:CreateLogStream",
        "logs:PutLogEvents"
      ],
      "Resource": "arn:aws:logs:*:*:*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "ec2:DescribeNetworkInterfaces",
        "ec2:CreateNetworkInterface",
        "ec2:DeleteNetworkInterface",
        "ec2:DescribeInstances",
        "ec2:AttachNetworkInterface"
      ],
      "Resource": "*"
    },
    {
       "Effect": "Allow",
       "Action": [
         "dynamodb:DescribeStream",
         "dynamodb:GetRecords",
         "dynamodb:GetShardIterator",
         "dynamodb:ListStreams"
       ],
       "Resource": "arn:aws:dynamodb:*:*:*"
    }
  ]
}
EOF
}

resource "aws_ecr_repository" "lambda_stream_repository" {
  name                 = "lambda-stream"
  image_tag_mutability = "MUTABLE"
}

resource "aws_lambda_function" "lambda_stream_function" {
  function_name = "lambda-stream"

  role = aws_iam_role.lambda_role.arn

  image_uri    = "${aws_ecr_repository.lambda_stream_repository.repository_url}:latest"
  package_type = "Image"

  vpc_config {
    subnet_ids         = module.vpc.private_subnets
    security_group_ids = [module.security_group.security_group_id]
  }

  timeout = 500
}

resource "aws_lambda_event_source_mapping" "lambda_stream_event_source" {
  event_source_arn  = aws_dynamodb_table.dynamodb.stream_arn
  function_name     = aws_lambda_function.lambda_stream_function.arn
  starting_position = "LATEST"
}
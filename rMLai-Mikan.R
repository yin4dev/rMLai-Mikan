app_name = "rMLai-Mikan v0.1.1"
# 20250309
# Hongrong Yin

library(shiny)
library(dplyr)
library(ggplot2)
library(skimr)
library(tidyr)
library(DT)
library(rhandsontable)
library(shinyBS)
library(glmnet)
library(GA)
library(randomForest)
library(rpart)
library(xgboost)
library(pROC)
library(e1071)  
library(kknn)   
library(nnet)  
library(shinyjs)
library(httr)
library(jsonlite)
library(pdftools)
library(readtext)
library(tools)

###############################################################################
# Fixed Parameters for Professional Data Analyst
###############################################################################
LLM_SYSTEM_PROMPT <- paste0(
  "You are a highly skilled professional data analyst. ",
  "Your responses must be precise, logical, and data-driven. ",
  "Provide clear insights, and statistical explanations while avoiding unnecessary speculation. ",
  "Always support your conclusions with numerical evidence."
)
LLM_TEMPERATURE <- 0.3
LLM_TOP_K <- 40
LLM_TOP_P <- 0.8
LLM_CONTEXT_WINDOW <- 2048  

###############################################################################
# Mapping of Internal Model Names to General Names
###############################################################################
MODEL_NAME_MAP <- c(
  # --- Regression ---
  "regr.rpart"      = "Decision tree (regr)",
  "regr.lm"         = "Linear regression (regr)",
  "regr.rf"         = "Random forests (regr)",
  "regr.xgboost"    = "xgboost (regr)",
  "regr.lasso_cv"   = "LASSO (regr)",
  # --- Classification ---
  "classif.rpart"      = "Decision tree (class)",
  "classif.rf"         = "Random forests (class)",
  "classif.xgboost"    = "xgboost (class)",
  "classif.svm"        = "svm (class)",
  "classif.log_reg"    = "Logistic regression (class)",
  "classif.naive_bayes"= "Naive Bayes (class)",
  "classif.kknn"       = "k-nearest neighbour (class)"
)

REGRESSION_CHOICES <- c(
  "Decision tree (regr)"   = "regr.rpart",
  "Linear regression (regr)" = "regr.lm",
  "Random forests (regr)"  = "regr.rf",
  "xgboost (regr)"         = "regr.xgboost",
  "LASSO (regr)"           = "regr.lasso_cv"
)

CLASSIFICATION_CHOICES <- c(
  "Decision tree (class)"        = "classif.rpart",
  "Random forests (class)"       = "classif.rf",
  "xgboost (class)"              = "classif.xgboost",
  "svm (class)"                  = "classif.svm",
  "Logistic regression (class)"  = "classif.log_reg",
  "Naive Bayes (class)"          = "classif.naive_bayes",
  "k-nearest neighbour (class)"  = "classif.kknn"
)

###############################################################################
# Helper Functions for Numerical and Categorical Association
###############################################################################
numeric_numeric_assoc <- function(x, y) {
  suppressWarnings(cor(x, y, use = "complete.obs"))
}
cramers_v <- function(x, y) {
  tbl <- table(x, y)
  chi2 <- suppressWarnings(chisq.test(tbl, correct = FALSE)$statistic)
  n <- sum(tbl)
  min_dim <- min(nrow(tbl), ncol(tbl))
  sqrt(as.numeric(chi2) / (n * (min_dim - 1)))
}
correlation_ratio <- function(factor_col, numeric_col) {
  f <- as.factor(factor_col)
  y <- numeric_col
  m <- mean(y)
  denominator <- sum((y - m)^2)
  numerator <- 0
  for (lvl in levels(f)) {
    y_l <- y[f == lvl]
    m_l <- mean(y_l)
    numerator <- numerator + length(y_l)*(m_l - m)^2
  }
  if (denominator == 0) 0 else sqrt(numerator / denominator)
}
measure_association <- function(x, y) {
  if(is.numeric(x) && is.numeric(y)){
    numeric_numeric_assoc(x, y)
  } else if(is.factor(x) && is.factor(y)){
    cramers_v(x, y)
  } else if(is.factor(x) && is.numeric(y)){
    correlation_ratio(x, y)
  } else if(is.numeric(x) && is.factor(y)){
    correlation_ratio(y, x)
  } else {
    NA_real_
  }
}
custom_box_plot <- function(df, variable, target) {
  ggplot(df, aes_string(x = variable, y = target)) +
    geom_boxplot(fill = "steelblue") +
    theme_minimal() +
    labs(x = variable, y = target, title = paste(target, "vs", variable))
}

###############################################################################
# get_var_importance_str Function
###############################################################################
get_var_importance_str <- function(model, iname, train_data, target_var, features) {
  if(grepl("rpart", iname)) {
    imp <- model$variable.importance
    if(is.null(imp)) return("No varImp")
    return(paste(paste(names(imp), round(imp,3), sep="="), collapse=", "))
  } else if(grepl("rf", iname)) {
    imp <- model$importance
    if(is.null(imp)) return("No varImp")
    if(is.matrix(imp)) {
      if("MeanDecreaseGini" %in% colnames(imp)) {
        imp_vals <- imp[, "MeanDecreaseGini"]
        return(paste(paste(rownames(imp), round(imp_vals,3), sep="="), collapse=", "))
      } else {
        imp_vals <- rowMeans(imp)
        return(paste(paste(rownames(imp), round(imp_vals,3), sep="="), collapse=", "))
      }
    } else {
      return(paste(paste(names(imp), round(imp,3), sep="="), collapse=", "))
    }
  } else if(grepl("xgboost", iname)) {
    bst <- model
    imp_df <- tryCatch(xgb.importance(feature_names=features, model=bst), error=function(e)NULL)
    if(is.null(imp_df)) return("No varImp")
    topv <- imp_df[, c("Feature", "Gain")]
    return(paste(apply(topv, 1, function(r) paste0(r[1],"=", round(as.numeric(r[2]),3))), collapse=", "))
  } else if(grepl("lasso_cv", iname)) {
    coefs <- coef(model, s="lambda.min")
    coefs_mat <- as.matrix(coefs)
    nm <- rownames(coefs_mat)
    vals <- round(coefs_mat[,1],3)
    return(paste(paste0(nm,"=", vals), collapse=", "))
  } else if(grepl("regr.lm", iname)) {
    coefs <- stats::coef(model)
    nm <- names(coefs)
    return(paste(paste(nm, round(coefs,3), sep="="), collapse=", "))
  } else if(grepl("svm", iname)) {
    return("Not Supported (svm)")
  } else if(grepl("naive_bayes", iname)) {
    return("Not Supported (naiveBayes)")
  } else if(grepl("kknn", iname)) {
    return("Not Supported (kknn)")
  } else if(grepl("log_reg", iname)) {
    return("Logistic coefs or Not Supported")
  }
  return("Not Supported")
}

###############################################################################
# Updated Classification Metrics Function with AUC Calculation
###############################################################################
get_classification_metrics <- function(y_true, y_pred, prob = NULL) {
  if(!is.factor(y_true) || !is.factor(y_pred)) 
    return(list(Accuracy = NA, F1 = NA, AUC = NA))
  
  acc <- mean(y_true == y_pred)
  
  # Only compute F1 and AUC for binary classification
  if(length(levels(y_true)) == 2){
    pos <- levels(y_true)[2]
    tp <- sum(y_pred == pos & y_true == pos)
    fp <- sum(y_pred == pos & y_true != pos)
    fn <- sum(y_pred != pos & y_true == pos)
    precision <- ifelse(tp+fp == 0, 0, tp/(tp+fp))
    recall    <- ifelse(tp+fn == 0, 0, tp/(tp+fn))
    f1        <- ifelse(precision+recall == 0, 0, 2*precision*recall/(precision+recall))
    
    auc_value <- NA
    if (!is.null(prob)) {
      # Use pROC package to compute AUC; 'prob' must be the predicted probability for the positive class.
      roc_obj <- try(pROC::roc(response = y_true, predictor = prob, levels = levels(y_true)), silent = TRUE)
      if (!inherits(roc_obj, "try-error")) {
        auc_value <- as.numeric(pROC::auc(roc_obj))
      }
    }
    
    return(list(Accuracy = acc, F1 = f1, AUC = auc_value))
  } else {
    return(list(Accuracy = acc, F1 = NA, AUC = NA))
  }
}

###############################################################################
# UI
###############################################################################
ui <- fluidPage(
  useShinyjs(),
  titlePanel(app_name), 
  
  sidebarLayout(
    sidebarPanel(
      fileInput("file", "Upload CSV File", accept = c("text/csv", ".csv")),
      uiOutput("name_var_ui"),
      uiOutput("target_var_ui"),
      uiOutput("target_var_type_ui"),
      uiOutput("features_var_ui"),
      uiOutput("features_type_ui"),
      width = 3
    ),
    
    mainPanel(
      tabsetPanel(
        tabPanel("Dataset", DT::dataTableOutput("filtered_data_table")),
        tabPanel("Distribution Visualization",
                 h4("Target Variable Distribution"), plotOutput("target_dist_plot"),
                 h4("Feature Variables Distribution"), uiOutput("features_dist_plots")
        ),
        tabPanel("Target vs Feature",
                 h4("Correlation Table"), tableOutput("target_feature_table"),
                 h4("Correlation Plots"), uiOutput("target_feature_corr_plots")
        ),
        tabPanel("Feature Association",
                 h4("Association Table"), tableOutput("features_assoc_table"),
                 h4("Association Heatmap"), plotOutput("features_assoc_heatmap")
        ),
        tabPanel("Similarity",
                 radioButtons("similarity_method", "Data Selection Method",
                              choices = c("Select by Name", "Manual Input"), inline = TRUE),
                 uiOutput("similarity_selection_ui"),
                 numericInput("candidate_num", "Number of Candidates", value = 5, min = 1),
                 actionButton("find_similarity", "Find Similar", class = "btn-primary"),
                 DT::dataTableOutput("similarity_table"),
                 plotOutput("similarity_plot") 
        ),
        tabPanel("Model Building",
                 selectizeInput("model_select", "Select Learning Model(s) (Multiple Allowed)",
                                choices = NULL, multiple = TRUE),
                 bsCollapse(
                   id = "hyperparam_collapse", open = NULL,
                   bsCollapsePanel(
                     title = "Hyperparameter Settings (Click to Expand)",
                     uiOutput("hyperparam_ui"), style = "default"
                   )
                 ),
                 actionButton("train_model", "Train Model", class = "btn-primary"),
                 br(), br(),
                 h4("Training Results (Metrics)"),
                 DT::dataTableOutput("model_metrics_table1"),
                 h4("Trained Model (Hyperparams, VarImp)"),
                 DT::dataTableOutput("model_metrics_table2"),
                 h4("Prediction Results on Existing Data (Plot)"),
                 plotOutput("prediction_plot"), br()
        ),
        tabPanel("Prediction",
                 h4("Prediction with New Data (Manual Input or CSV)"), br(),
                 selectInput("model_for_prediction", "Model for Prediction", choices = NULL),
                 actionButton("predict_new", "Run Prediction", class = "btn-success"), br(),
                 downloadButton("download_pred", "Download Prediction Results"),
                 radioButtons("newdata_input_method", "Input Method for Prediction Data",
                              choices = c("Upload CSV", "Manual Input"), selected = "Manual Input", inline = TRUE),
                 fileInput("newdata_file", "Upload New Prediction CSV", accept = c("text/csv", ".csv")),
                 rHandsontableOutput("new_data_table"),
                 br()
        ),
        tabPanel("Inverse Prediction",
                 h4("Inverse Prediction Parameter Settings"),
                 selectInput("inverse_model", "Select Model to Use", choices = NULL),
                 uiOutput("target_value_ui"),
                 numericInput("n_solutions", "Number of Optimal Solutions", value = 5, min = 1),
                 uiOutput("var_range_ui"),
                 actionButton("run_ga", "Run Genetic Algorithm", class = "btn-warning"),
                 h4("Optimal Solution Candidates"),
                 DT::dataTableOutput("ga_results_table")
        ),
        tabPanel("Report",
                 actionButton("generate_report", "Generate/Update Report"),
                 downloadButton("download_report", "Download Report (txt)"),
                 br(), br(),
                 verbatimTextOutput("report_output")
        ),
        tabPanel("AI Assistant",
                 selectInput("model", "Select LLM Model", 
                             choices = c("llama3.2:3b", 
                                         "deepseek-r1:8b")),
                 actionButton("newChat", "New Chat"),
                 downloadButton("downloadHistory", "Download Chat History"),
                 # Chat Display
                 div(
                   style = "border: 1px solid #ccc; padding: 10px; background-color: #f9f9f9; height: 400px; overflow-y: auto;",
                   uiOutput("chatHistoryUI")
                 ),
                 div(id = "loadingIndicator", 
                     style = "display:none; font-weight:bold; color:blue; margin-bottom: 10px;", 
                     "LLM is responding..."),
                 
                 fluidRow(
                   column(
                     width = 4,
                     fileInput("UserFile", "Upload File", multiple = TRUE, 
                               accept = c(".txt", ".csv", ".pdf", ".doc", ".docx"))
                   ),
                   column(
                     width = 4,
                     selectInput("historyFilesSelect", "Select History File(s):", choices = c(), multiple = TRUE)
                   )
                 ),
                 textAreaInput("userInput", "Message", value = "", 
                               placeholder = "Enter your message here...", 
                               rows = 3, width = "100%", resize = "both"),
                 
                 fluidRow(
                   column(width = 2, actionButton("sendMsg", "Send", class = "btn-primary")),
                   column(width = 5, actionButton("sendReport2LLM", "Attach Report")),
                   column(width = 5, checkboxInput("suppressChatHistory", "One-time Query", value = FALSE))
                 )
        ),
        tabPanel("Model Explanation",
                 uiOutput("model_explanation")
        )
      )
    )
  )
)

###############################################################################
# SERVER
###############################################################################
server <- function(input, output, session) {
  
  #=========================== AI Tab =============================#
  
  newChatTrigger <- reactiveVal(0)         
  
  uploadedFiles <- reactiveVal(list())
  newUploadedFiles <- reactiveVal(list())
  
  chatHistory <- reactiveVal(list())
  
  addMessage <- function(role, content) {
    current <- chatHistory()
    newMsg <- list(
      role = role, 
      content = content,
      timestamp = Sys.time()
    )
    chatHistory(append(current, list(newMsg)))
  }
  
  output$chatHistoryUI <- renderUI({
    msgs <- chatHistory()
    bubbleList <- lapply(msgs, function(msg) {
      bubbleClass <- if (msg$role == "user") {
        "padding:8px; margin:4px; border-radius:8px; background-color:#007BFF; color:white; align-self:flex-end;"
      } else {
        "padding:8px; margin:4px; border-radius:8px; background-color:#E9ECEF; color:black; align-self:flex-start;"
      }
      div(style = bubbleClass,
          div(style = "font-size:0.8em; color:#666;",
              format(msg$timestamp, "%Y-%m-%d %H:%M:%S")),
          HTML(gsub("\n", "<br>", msg$content))
      )
    })
    do.call(tagList, bubbleList)
  })
  
  observeEvent(input$newChat, {
    chatHistory(list())
    uploadedFiles(list())
    updateSelectInput(session, "historyFilesSelect", choices = character(0))
    newChatTrigger(newChatTrigger() + 1)
    showNotification("Chat history and context have been cleared. Starting a new conversation.", type = "message")
  })
  
  multiFileContents <- reactive({
    if (is.null(input$UserFile) || nrow(input$UserFile) == 0) {
      return(list())
    }
    allFiles <- list()
    for (i in seq_len(nrow(input$UserFile))) {
      fn <- input$UserFile$name[i]
      dp <- input$UserFile$datapath[i]
      ext <- tolower(file_ext(fn))
      content <- ""
      
      if (!ext %in% c("txt", "csv", "pdf", "doc", "docx")) {
        showNotification(paste("Unsupported file type:", fn), type = "error")
        next
      }
      
      if (ext %in% c("txt", "csv")) {
        content <- tryCatch(paste(readLines(dp, warn = FALSE), collapse = "\n"),
                            error = function(e) "")
      } else if (ext == "pdf") {
        content <- tryCatch(paste(pdftools::pdf_text(dp), collapse = "\n"),
                            error = function(e) "")
      } else if (ext %in% c("doc", "docx")) {
        dt <- tryCatch(readtext::readtext(dp), error = function(e) NULL)
        if (!is.null(dt)) {
          content <- paste(dt$text, collapse = "\n")
        }
      }
      
      allFiles[[fn]] <- content
    }
    allFiles
  })
  
  observeEvent(input$UserFile, {
    newUploadedFiles(multiFileContents())
  })
  
  observeEvent(input$assistantProfile, {
    selectedProfile <- assistantProfiles[[ input$assistantProfile ]]
    updateSliderInput(session, "temp", value = selectedProfile$temp)
    updateNumericInput(session, "topK", value = selectedProfile$topK)
    updateNumericInput(session, "topP", value = selectedProfile$topP)
    updateTextAreaInput(session, "systemPrompt", value = selectedProfile$system)
  })
  
  observeEvent(input$sendMsg, {
    req(input$userInput)
    
    userMsg <- input$userInput
    newFilesContent <- newUploadedFiles()
    newFiles <- names(newFilesContent)
    
    appendedLine <- character(0)
    
    if (length(newFiles) > 0) {
      appendedLine <- c(appendedLine, 
                        paste0("(Uploaded File(s): ", paste(newFiles, collapse = ", "), ")"))
    }
    
    if (length(input$historyFilesSelect) > 0) {
      appendedLine <- c(appendedLine,
                        paste0("(History File(s): ",
                               paste(input$historyFilesSelect, collapse = ", "),
                               ")"))
    }
    
    if (length(appendedLine) > 0) {
      userMsg <- paste(userMsg, paste(appendedLine, collapse = " "), sep = "\n")
    }
    
    shinyjs::disable("sendMsg")
    shinyjs::show("loadingIndicator")
    
    addMessage("user", userMsg)
    
    historyText <- sapply(chatHistory(), function(x) {
      paste0(if(x$role == "user") "User: " else "Assistant: ", x$content)
    }, USE.NAMES = FALSE)
    
    combinedPrompt <- paste("\n [SYSTEM PROMPT] \n", input$systemPrompt)
    
    if (input$suppressChatHistory) {
      combinedPrompt <- paste(
        combinedPrompt, 
        "\n\n [LATEST USER PROMPT] \n", 
        input$userInput
      )
    } else {
      historyText <- sapply(chatHistory(), function(x) {
        paste0(if (x$role == "user") "User: " else "Assistant: ", x$content)
      }, USE.NAMES = FALSE)
      combinedPrompt <- paste(
        combinedPrompt, 
        "\n\n [CHAT HISTORY] \n", 
        paste(historyText, collapse = "\n")
      )
    }
    
    if (length(newFiles) > 0) {
      combinedPrompt <- paste(combinedPrompt, "\n\n [FILE CONTENT] \n")
      for (fn in newFiles) {
        text <- newFilesContent[[fn]]
        combinedPrompt <- paste0(
          combinedPrompt, 
          "-----\n", fn, ":\n", text, "\n"
        )
      }
    }
    
    `%||%` <- function(x, y) {
      if (!is.null(x)) x else y
    }
    
    if (length(input$historyFilesSelect) > 0) {
      uf <- uploadedFiles()
      blocks <- lapply(input$historyFilesSelect, function(fn) {
        uf[[fn]] %||% ""
      })
      histBlock <- paste(blocks, collapse = "\n---\n")
      combinedPrompt <- paste(combinedPrompt, "\n\n [HISTORY FILES] \n", histBlock)
    }
    
    reqBody <- list(
      model = input$model,
      prompt = combinedPrompt,
      temperature = input$temp,
      context_window = input$context,
      top_k = input$topK,
      top_p = input$topP
    )
    
    fullResponse <- tryCatch({
      res <- POST("http://localhost:11434/api/generate", 
                  body = toJSON(reqBody, auto_unbox = TRUE), 
                  encode = "json")
      resText <- content(res, "text", encoding = "UTF-8")
      lines <- strsplit(resText, "\n")[[1]]
      responseParts <- sapply(lines, function(line) {
        if(nchar(trimws(line)) > 0) {
          parsed <- fromJSON(line)
          return(parsed$response)
        } else {
          return("")
        }
      })
      paste(responseParts, collapse = "")
    }, error = function(e) {
      paste("Error during API call:", e$message)
    })
    
    addMessage("bot", fullResponse)
    
    if (length(newFiles) > 0) {
      curr <- uploadedFiles()
      for (fn in newFiles) {
        curr[[fn]] <- newFilesContent[[fn]]
      }
      uploadedFiles(curr)
      updateSelectInput(session, "historyFilesSelect", 
                        choices = names(curr),
                        selected = input$historyFilesSelect)
      newUploadedFiles(list())
    }
    
    updateTextAreaInput(session, "userInput", value = "")
    reset("UserFile")
    
    shinyjs::enable("sendMsg")
    shinyjs::hide("loadingIndicator")
  })
  
  output$downloadHistory <- downloadHandler(
    filename = function() {
      paste0("chat_history_", Sys.Date(), ".txt")
    },
    content = function(file) {
      msgs <- chatHistory()
      lines <- sapply(msgs, function(msg) {
        paste0(
          "[", format(msg$timestamp, "%Y-%m-%d %H:%M:%S"), "] ",
          if(msg$role == "user") "User: " else "Assistant: ",
          msg$content
        )
      })
      writeLines(lines, con = file)
    }
  )
  
  observeEvent(once = TRUE, TRUE, {
    newChatTrigger(newChatTrigger() + 1)
  })
  
  # Send Report to LLM
  observeEvent(input$sendReport2LLM, {
    rep_txt <- reportContent()
    if(nchar(rep_txt) < 10) {
      showNotification("Report content is empty. Please generate the report first.", type="error")
      return()
    }
    # Store in uploadedFiles
    uf <- uploadedFiles()
    uf[["MachineLearning_Report.txt"]] <- rep_txt
    uploadedFiles(uf)
    updateSelectInput(session, "historyFilesSelect",
                      choices = names(uf),
                      selected = c(input$historyFilesSelect, "MachineLearning_Report.txt"))
    showNotification("The machine learning report has been incorporated into the LLM.", type = "message")
  })
  
  #======================= Machine Learning Part ======================#
  
  # CSV reading
  data_reactive <- reactive({
    if(is.null(input$file)) {
      return(NULL)
    }
    df <- tryCatch({
      read.csv(input$file$datapath, header=TRUE, stringsAsFactors=FALSE)
    }, error = function(e){
      showNotification(paste("CSV reading error:", e$message), type="error")
      return(NULL)
    })
    if(is.null(df)) return(NULL)
    colnames(df) <- make.names(colnames(df), unique=TRUE)
    df
  })
  
  # UI for Target and Feature Variables
  observeEvent(data_reactive(), {
    df <- data_reactive()
    if(is.null(df)) {
      return()
    }
    vars <- names(df)
    output$target_var_ui <- renderUI({
      selectInput("target_var", "Target Variable", choices = vars, selected = vars[1])
    })
    output$target_var_type_ui <- renderUI({
      if(is.null(input$target_var)) return(NULL)
      default_type <- if(is.numeric(df[[input$target_var]])) "Numeric" else "Factor"
      radioButtons("target_var_type", "Type of Target Variable", choices=c("Numeric", "Factor"),
                   selected=default_type, inline=TRUE)
    })
    output$features_var_ui <- renderUI({
      if(is.null(input$target_var)) return(NULL)
      remaining_vars <- setdiff(vars, input$target_var)
      selectizeInput("features_var", "Feature Variables",
                     choices = remaining_vars, multiple=TRUE, selected=NULL)
    })
  })
  
  observeEvent(input$features_var, {
    if(is.null(input$features_var)) {
      output$features_type_ui <- renderUI({NULL})
      return()
    }
    df <- data_reactive()
    output$features_type_ui <- renderUI({
      lapply(input$features_var, function(f) {
        default_type <- if(is.numeric(df[[f]])) "Numeric" else "Factor"
        radioButtons(paste0("feature_type_", f),
                     label=paste("Type for", f),
                     choices=c("Numeric", "Factor"),
                     selected=default_type,
                     inline=TRUE)
      }) %>% tagList()
    })
  })
  
  output$name_var_ui <- renderUI({
    req(data_reactive())
    df <- data_reactive()
    selectInput("name_var", "Name Variable", choices = names(df))
  })
  
  # Determine Task Type
  task_type <- reactive({
    if(is.null(input$target_var_type)) {
      return(NULL)
    }
    if(input$target_var_type == "Numeric") "Regression" else "Classification"
  })
  
  # Update Model Choices
  observe({
    if(is.null(task_type())) return()
    if(task_type() == "Regression"){
      updateSelectizeInput(session, "model_select", choices=REGRESSION_CHOICES, selected=NULL)
    } else {
      choices <- CLASSIFICATION_CHOICES
      updateSelectizeInput(session, "model_select", choices=choices, selected=NULL)
    }
  })
  
  # Remove NAs and Type Conversion
  get_filtered_data <- reactive({
    df <- data_reactive()
    if(is.null(df)) {
      return(NULL)
    }
    if(is.null(input$target_var) || input$target_var=="") {
      return(NULL)
    }
    if(is.null(input$target_var_type)) {
      return(NULL)
    }
    
    # Target variable type
    if(input$target_var_type == "Numeric") {
      df[[input$target_var]] <- suppressWarnings(as.numeric(as.character(df[[input$target_var]])))
    } else {
      df[[input$target_var]] <- factor(df[[input$target_var]])
    }
    
    # Feature variables type
    if(is.null(input$features_var)) {
      return(NULL)
    }
    for(f in input$features_var) {
      ft <- input[[paste0("feature_type_", f)]]
      if(!is.null(ft)) {
        if(ft=="Numeric") {
          df[[f]] <- suppressWarnings(as.numeric(as.character(df[[f]])))
        } else {
          df[[f]] <- factor(df[[f]])
        }
      }
    }
    sel_cols <- c(input$target_var, input$features_var)
    df2 <- df[complete.cases(df[, sel_cols, drop=FALSE]), ]
    
    if(nrow(df2) == 0) {
      showNotification("No rows in the filtered data. Please review type settings or numeric conversions.", type="error")
      return(NULL)
    }
    df2[, sel_cols, drop=FALSE]
  })
  
  # Data Display and Visualization
  output$filtered_data_table <- DT::renderDataTable({
    df <- get_filtered_data()
    if(is.null(df)) {
      return(data.frame(Message="No data available."))
    }
    if(!is.null(input$name_var) && input$name_var %in% names(df)){
      df <- df[c(input$name_var, setdiff(names(df), input$name_var))]
    }
    df
  }, options=list(scrollX=TRUE, pageLength=10))
  
  output$target_dist_plot <- renderPlot({
    df <- get_filtered_data()
    if(is.null(df) || nrow(df)==0) {
      showNotification("No data available for visualization.", type="error")
      return()
    }
    target_col <- input$target_var
    if(is.numeric(df[[target_col]])) {
      ggplot(df, aes_string(x=target_col)) +
        geom_histogram(fill="steelblue", color="white") +
        theme_minimal() +
        labs(x=target_col, y="Count", title=paste("Histogram of", target_col))
    } else {
      ggplot(df, aes_string(x=target_col)) +
        geom_bar(fill="steelblue") +
        theme_minimal() +
        labs(x=target_col, y="Count", title=paste("Barplot of", target_col))
    }
  })
  
  output$features_dist_plots <- renderUI({
    df <- get_filtered_data()
    if(is.null(df) || nrow(df)==0) {
      return(tags$p("No data available for visualization.", style="color:red;"))
    }
    if(is.null(input$features_var)) return(NULL)
    
    plot_outputs <- lapply(input$features_var, function(varname){
      plotname <- paste0("dist_plot_", varname)
      output[[plotname]] <- renderPlot({
        if(is.numeric(df[[varname]])) {
          ggplot(df, aes_string(x=varname)) +
            geom_histogram(fill="orange", color="white") +
            theme_minimal() +
            labs(x=varname, y="Count", title=paste("Histogram of", varname))
        } else {
          ggplot(df, aes_string(x=varname)) +
            geom_bar(fill="orange") +
            theme_minimal() +
            labs(x=varname, y="Count", title=paste("Barplot of", varname))
        }
      })
      plotOutput(plotname, height="250px")
    })
    fluidRow(lapply(seq_along(plot_outputs), function(i){
      column(width=6, plot_outputs[[i]])
    }))
  })
  
  output$target_feature_table <- renderTable({
    df <- get_filtered_data()
    if(is.null(df) || nrow(df)==0) {
      return(data.frame(Message="No data available."))
    }
    if(is.null(input$features_var)) {
      return(data.frame(Message="No feature variables selected."))
    }
    assoc_vec <- sapply(input$features_var, function(f){
      val <- measure_association(df[[input$target_var]], df[[f]])
      round(val, 3)
    })
    as.data.frame(assoc_vec)
  }, rownames=TRUE)
  
  output$target_feature_corr_plots <- renderUI({
    df <- get_filtered_data()
    if(is.null(df) || nrow(df)==0) {
      return(tags$p("No data available.", style="color:red;"))
    }
    if(is.null(input$features_var)) return(NULL)
    
    target <- input$target_var
    plot_outputs <- lapply(input$features_var, function(fea){
      plotname <- paste0("corr_plot_", fea)
      output[[plotname]] <- renderPlot({
        x <- df[[fea]]
        y <- df[[target]]
        if(is.numeric(x) && is.numeric(y)) {
          corval <- suppressWarnings(cor(x, y))
          ggplot(df, aes_string(x=fea, y=target)) +
            geom_point(alpha=0.6) +
            geom_smooth(method="lm", se=FALSE, color="red") +
            labs(title=paste(target, "vs", fea),
                 subtitle=paste("Pearson r =", round(corval,3))) +
            theme_minimal()
        } else if(is.factor(x) && is.numeric(y)) {
          custom_box_plot(df, fea, target)
        } else if(is.numeric(x) && is.factor(y)) {
          custom_box_plot(df, target, fea)
        } else if(is.factor(x) && is.factor(y)) {
          ggplot(df, aes_string(x=fea, fill=target)) +
            geom_bar(position="dodge") +
            theme_minimal() +
            labs(title=paste(target, "vs", fea))
        } else {
          plot.new()
          text(0.5, 0.5, "No corresponding plot available.")
        }
      })
      plotOutput(plotname, height="300px")
    })
    fluidRow(lapply(seq_along(plot_outputs), function(i){
      column(width=6, plot_outputs[[i]])
    }))
  })
  
  output$features_assoc_table <- renderTable({
    df <- get_filtered_data()
    if(is.null(df) || nrow(df)==0) {
      return(data.frame(Message="No data available."))
    }
    if(is.null(input$features_var) || length(input$features_var)<2){
      return(data.frame(Message="At least 2 feature variables are required."))
    }
    assoc_mat <- matrix(NA, nrow=length(input$features_var), ncol=length(input$features_var),
                        dimnames=list(input$features_var, input$features_var))
    for(i in seq_along(input$features_var)){
      for(j in seq_along(input$features_var)){
        assoc_mat[i,j] <- measure_association(df[[input$features_var[i]]], df[[input$features_var[j]]])
      }
    }
    assoc_mat <- round(assoc_mat,3)
    as.data.frame(assoc_mat)
  }, rownames=TRUE)
  
  output$features_assoc_heatmap <- renderPlot({
    df <- get_filtered_data()
    if(is.null(df) || nrow(df)==0){
      showNotification("No data available.", type="error")
      return()
    }
    if(is.null(input$features_var) || length(input$features_var)<2){
      plot.new()
      text(0.5, 0.5, "At least 2 feature variables are required.")
      return()
    }
    assoc_mat <- matrix(NA, nrow=length(input$features_var), ncol=length(input$features_var),
                        dimnames=list(input$features_var, input$features_var))
    for(i in seq_along(input$features_var)){
      for(j in seq_along(input$features_var)){
        assoc_mat[i,j] <- measure_association(df[[input$features_var[i]]], df[[input$features_var[j]]])
      }
    }
    assoc_mat <- round(assoc_mat,3)
    assoc_df <- as.data.frame(assoc_mat)
    assoc_df$Var1 <- rownames(assoc_mat)
    melted <- pivot_longer(assoc_df, cols=-Var1, names_to="Var2", values_to="Association")
    ggplot(melted, aes(x=Var1, y=Var2, fill=Association)) +
      geom_tile(color="white") +
      scale_fill_gradient2(low="blue", high="red", mid="white", midpoint=0,
                           limits=c(-1,1), na.value="grey80") +
      theme_minimal() +
      theme(axis.text.x=element_text(angle=45, hjust=1)) +
      labs(x="", y="", fill="Assoc", title="Features Association Heatmap")
  })
  
  # Model Training and Evaluation
  trained_models <- reactiveVal(list())
  model_metrics_data <- reactiveVal(NULL)
  
  get_regression_metrics <- function(y_true, y_pred) {
    mae <- mean(abs(y_true - y_pred))
    mse <- mean((y_true - y_pred)^2)
    rmse <- sqrt(mse)
    r2 <- 1 - sum((y_true - y_pred)^2)/sum((y_true - mean(y_true))^2)
    list(MAE=mae, MSE=mse, RMSE=rmse, R2=r2)
  }
  
  output$hyperparam_ui <- renderUI({
    if(is.null(input$model_select)) return(NULL)
    df <- data_reactive()
    if(is.null(df) || is.null(input$target_var)) return(NULL)
    
    if(task_type()=="Regression"){
      lapply(input$model_select, function(iname){
        switch(iname,
               "regr.rpart"=tagList(
                 h5("Decision tree (regr)"),
                 numericInput("regr_rpart_cp", "cp", 0.01, step=0.001)
               ),
               "regr.lm"=tagList(
                 h5("Linear regression (regr): No parameters")
               ),
               "regr.rf"=tagList(
                 h5("Random forests (regr)"),
                 numericInput("regr_rf_mtry", "mtry", 2, min=1, step=1),
                 numericInput("regr_rf_ntree", "ntree", 500, min=1, step=50)
               ),
               "regr.xgboost"=tagList(
                 h5("xgboost (regr)"),
                 numericInput("regr_xgb_eta", "eta", 0.1, min=0.01, step=0.01),
                 numericInput("regr_xgb_max_depth", "max_depth", 6, min=1, step=1),
                 numericInput("regr_xgb_nrounds", "nrounds", 100, min=1, step=10)
               ),
               "regr.lasso_cv"=tagList(
                 h5("LASSO (regr)"),
                 numericInput("regr_lasso_cv_nfolds", "nfolds", 5, min=3, step=1)
               )
        )
      })
    } else {
      lapply(input$model_select, function(iname){
        switch(iname,
               "classif.rpart"=tagList(
                 h5("Decision tree (class)"),
                 numericInput("classif_rpart_cp", "cp", 0.01, step=0.001)
               ),
               "classif.rf"=tagList(
                 h5("Random forests (class)"),
                 numericInput("classif_rf_mtry", "mtry", 2, min=1, step=1),
                 numericInput("classif_rf_ntree", "ntree", 500, min=1, step=50)
               ),
               "classif.xgboost"=tagList(
                 h5("xgboost (class) - 2-class only"),
                 numericInput("classif_xgb_eta", "eta", 0.1, min=0.01, step=0.01),
                 numericInput("classif_xgb_max_depth", "max_depth", 6, min=1, step=1),
                 numericInput("classif_xgb_nrounds", "nrounds", 100, min=1, step=10)
               ),
               "classif.svm"=tagList(
                 h5("SVM (class) - e1071::svm"),
                 selectInput("classif_svm_kernel", "kernel", 
                             choices=c("radial","linear","polynomial","sigmoid")),
                 numericInput("classif_svm_cost", "cost", 1, min=0.1, step=0.1),
                 numericInput("classif_svm_gamma", "gamma", 0.1, min=1e-4, step=0.001)
               ),
               "classif.log_reg"=tagList(
                 h5("Logistic regression (class)"),
                 p("2-class => glm. Multi-class => nnet::multinom")
               ),
               "classif.naive_bayes"=tagList(
                 h5("Naive Bayes (class) - e1071::naiveBayes")
               ),
               "classif.kknn"=tagList(
                 h5("k-nearest neighbour (class) - kknn"),
                 numericInput("classif_kknn_k", "k", 5, min=1, step=1)
               )
        )
      })
    }
  })
  
  # ------------------- Model Training -------------------
  observeEvent(input$train_model, {
    if(is.null(get_filtered_data())){
      showNotification("No training data available. Please upload and configure the CSV first.", type="error")
      return()
    }
    if(is.null(input$model_select) || length(input$model_select)==0){
      showNotification("Please select a learning model.", type="error")
      return()
    }
    if(is.null(input$target_var) || input$target_var==""){
      showNotification("Please select a target variable.", type="error")
      return()
    }
    
    withProgress(message="Training model(s)...", value=0, {
      df <- get_filtered_data()
      if(is.null(df) || nrow(df)==0){
        showNotification("Training data does not exist.", type="error")
        return()
      }
      
      target_col <- input$target_var
      feature_cols <- input$features_var
      actual <- df[[target_col]]
      
      current_models <- list()
      all_metrics <- list()
      predictions_list <- list()
      
      model_list <- as.character(input$model_select)
      
      # ---------------- Regression or Classification ----------------
      if(task_type()=="Regression"){
        # ---- Regression ----
        for(mod_iname in model_list){
          incProgress(1/length(model_list), detail=paste("Training:", MODEL_NAME_MAP[mod_iname]))
          
          if(mod_iname=="regr.rpart"){
            cp_val <- ifelse(is.null(input$regr_rpart_cp), 0.01, input$regr_rpart_cp)
            f <- as.formula(paste(target_col, "~", paste(feature_cols, collapse="+")))
            fit <- rpart::rpart(
              formula=f, data=df, method="anova",
              control=rpart::rpart.control(cp=cp_val)
            )
            pred <- predict(fit, newdata=df)
            mets <- get_regression_metrics(actual, pred)
            varimp <- get_var_importance_str(fit, mod_iname, df, target_col, feature_cols)
            
            current_models[[mod_iname]] <- fit
            predictions_list[[mod_iname]] <- pred
            all_metrics[[mod_iname]] <- list(
              ModelName=MODEL_NAME_MAP[mod_iname],
              MAE=round(mets$MAE,3),
              MSE=round(mets$MSE,3),
              RMSE=round(mets$RMSE,3),
              R2=round(mets$R2,3),
              Accuracy=NA, F1=NA, AUC=NA,
              Hyperparams=paste0("cp=",cp_val),
              VarImp=varimp
            )
            
          } else if(mod_iname=="regr.lm"){
            f <- as.formula(paste(target_col, "~", paste(feature_cols, collapse="+")))
            fit <- lm(f, data=df)
            pred <- predict(fit, newdata=df)
            mets <- get_regression_metrics(actual, pred)
            varimp <- get_var_importance_str(fit, mod_iname, df, target_col, feature_cols)
            
            current_models[[mod_iname]] <- fit
            predictions_list[[mod_iname]] <- pred
            all_metrics[[mod_iname]] <- list(
              ModelName=MODEL_NAME_MAP[mod_iname],
              MAE=round(mets$MAE,3),
              MSE=round(mets$MSE,3),
              RMSE=round(mets$RMSE,3),
              R2=round(mets$R2,3),
              Accuracy=NA, F1=NA, AUC=NA,
              Hyperparams="(no params)",
              VarImp=varimp
            )
            
          } else if(mod_iname=="regr.rf"){
            mtry_val <- ifelse(is.null(input$regr_rf_mtry), 2, input$regr_rf_mtry)
            ntree_val <- ifelse(is.null(input$regr_rf_ntree), 500, input$regr_rf_ntree)
            f <- as.formula(paste(target_col, "~", paste(feature_cols, collapse="+")))
            fit <- randomForest::randomForest(f, data=df, ntree=ntree_val, mtry=mtry_val)
            pred <- predict(fit, newdata=df)
            mets <- get_regression_metrics(actual, pred)
            varimp <- get_var_importance_str(fit, mod_iname, df, target_col, feature_cols)
            
            current_models[[mod_iname]] <- fit
            predictions_list[[mod_iname]] <- pred
            all_metrics[[mod_iname]] <- list(
              ModelName=MODEL_NAME_MAP[mod_iname],
              MAE=round(mets$MAE,3),
              MSE=round(mets$MSE,3),
              RMSE=round(mets$RMSE,3),
              R2=round(mets$R2,3),
              Accuracy=NA, F1=NA, AUC=NA,
              Hyperparams=paste0("mtry=",mtry_val,", ntree=",ntree_val),
              VarImp=varimp
            )
            
          } else if(mod_iname=="regr.xgboost"){
            eta_val <- ifelse(is.null(input$regr_xgb_eta), 0.1, input$regr_xgb_eta)
            md_val <- ifelse(is.null(input$regr_xgb_max_depth), 6, input$regr_xgb_max_depth)
            nr_val <- ifelse(is.null(input$regr_xgb_nrounds), 100, input$regr_xgb_nrounds)
            
            f <- as.formula(paste("~", paste(feature_cols, collapse="+")))
            X <- model.matrix(f, df)[, -1, drop=FALSE]
            y <- df[[target_col]]
            
            dtrain <- xgb.DMatrix(X, label=y)
            params <- list(objective="reg:squarederror", eta=eta_val, max_depth=md_val)
            fit <- xgb.train(params=params, data=dtrain, nrounds=nr_val, verbose=0)
            pred <- predict(fit, newdata=X)
            mets <- get_regression_metrics(actual, pred)
            varimp <- get_var_importance_str(fit, mod_iname, df, target_col, colnames(X))
            
            current_models[[mod_iname]] <- fit
            predictions_list[[mod_iname]] <- pred
            all_metrics[[mod_iname]] <- list(
              ModelName=MODEL_NAME_MAP[mod_iname],
              MAE=round(mets$MAE,3),
              MSE=round(mets$MSE,3),
              RMSE=round(mets$RMSE,3),
              R2=round(mets$R2,3),
              Accuracy=NA, F1=NA, AUC=NA,
              Hyperparams=paste0("eta=",eta_val,", max_depth=",md_val,", nrounds=",nr_val),
              VarImp=varimp
            )
            
          } else if(mod_iname=="regr.lasso_cv"){
            f <- as.formula(paste(target_col, "~", paste(feature_cols, collapse="+")))
            X <- model.matrix(f, df)[, -1, drop=FALSE]
            y <- df[[target_col]]
            nfolds_val <- ifelse(is.null(input$regr_lasso_cv_nfolds), 5, input$regr_lasso_cv_nfolds)
            
            fit <- cv.glmnet(X, y, alpha=1, nfolds=nfolds_val)
            pred <- predict(fit, newx=X, s="lambda.min")
            pred <- as.numeric(pred)
            mets <- get_regression_metrics(actual, pred)
            varimp <- get_var_importance_str(fit, mod_iname, df, target_col, colnames(X))
            
            current_models[[mod_iname]] <- fit
            predictions_list[[mod_iname]] <- pred
            all_metrics[[mod_iname]] <- list(
              ModelName=MODEL_NAME_MAP[mod_iname],
              MAE=round(mets$MAE,3),
              MSE=round(mets$MSE,3),
              RMSE=round(mets$RMSE,3),
              R2=round(mets$R2,3),
              Accuracy=NA, F1=NA, AUC=NA,
              Hyperparams=paste0("nfolds=",nfolds_val),
              VarImp=varimp
            )
          }
        }
        
        # Regression Results Visualization
        df_pred <- data.frame(actual=actual)
        for(nm in names(predictions_list)){
          df_pred[[nm]] <- predictions_list[[nm]]
        }
        df_long <- pivot_longer(df_pred, cols=-actual, names_to="model", values_to="predicted")
        p <- ggplot(df_long, aes(x=actual, y=predicted, color=model)) +
          geom_point(alpha=0.6) +
          geom_abline(slope=1, intercept=0, linetype="dashed") +
          theme_minimal() +
          labs(title="Regression: Actual vs Predicted", x="Actual", y="Predicted")
        output$prediction_plot <- renderPlot({ p })
        
      } else {
        # ---- Classification ----
        for(mod_iname in model_list){
          incProgress(1/length(model_list), detail=paste("Training:", MODEL_NAME_MAP[mod_iname]))
          
          if(mod_iname=="classif.rpart"){
            cp_val <- ifelse(is.null(input$classif_rpart_cp), 0.01, input$classif_rpart_cp)
            f <- as.formula(paste(target_col, "~", paste(feature_cols, collapse="+")))
            fit <- rpart::rpart(f, data=df, method="class",
                                control=rpart::rpart.control(cp=cp_val))
            pred <- predict(fit, newdata=df, type="class")
            # For AUC, probability estimates are needed; here we assume using the second class probability if available.
            prob <- tryCatch({
              p <- predict(fit, newdata=df)[,2]
              p
            }, error = function(e) NA)
            mets <- get_classification_metrics(actual, pred, prob)
            varimp <- get_var_importance_str(fit, mod_iname, df, target_col, feature_cols)
            
            current_models[[mod_iname]] <- fit
            predictions_list[[mod_iname]] <- pred
            all_metrics[[mod_iname]] <- list(
              ModelName=MODEL_NAME_MAP[mod_iname],
              MAE=NA, MSE=NA, RMSE=NA, R2=NA,
              Accuracy=round(mets$Accuracy,3),
              F1=round(mets$F1,3),
              AUC=round(mets$AUC,3),
              Hyperparams=paste0("cp=",cp_val),
              VarImp=varimp
            )
            
          } else if(mod_iname=="classif.rf"){
            mtry_val <- ifelse(is.null(input$classif_rf_mtry), 2, input$classif_rf_mtry)
            ntree_val <- ifelse(is.null(input$classif_rf_ntree), 500, input$classif_rf_ntree)
            f <- as.formula(paste(target_col, "~", paste(feature_cols, collapse="+")))
            fit <- randomForest::randomForest(f, data=df, ntree=ntree_val, mtry=mtry_val)
            pred <- predict(fit, newdata=df)
            prob <- tryCatch({
              p <- predict(fit, newdata=df, type="prob")[,2]
              p
            }, error = function(e) NA)
            mets <- get_classification_metrics(actual, pred, prob)
            varimp <- get_var_importance_str(fit, mod_iname, df, target_col, feature_cols)
            
            current_models[[mod_iname]] <- fit
            predictions_list[[mod_iname]] <- pred
            all_metrics[[mod_iname]] <- list(
              ModelName=MODEL_NAME_MAP[mod_iname],
              MAE=NA, MSE=NA, RMSE=NA, R2=NA,
              Accuracy=round(mets$Accuracy,3),
              F1=round(mets$F1,3),
              AUC=round(mets$AUC,3),
              Hyperparams=paste0("mtry=",mtry_val,", ntree=",ntree_val),
              VarImp=varimp
            )
            
          } else if(mod_iname=="classif.xgboost"){
            eta_val <- ifelse(is.null(input$classif_xgb_eta), 0.1, input$classif_xgb_eta)
            md_val <- ifelse(is.null(input$classif_xgb_max_depth), 6, input$classif_xgb_max_depth)
            nr_val <- ifelse(is.null(input$classif_xgb_nrounds), 100, input$classif_xgb_nrounds)
            
            lv <- levels(actual)
            if(length(lv)!=2){
              showNotification("xgboost (class) supports only 2 classes.", type="error")
              next
            }
            y_num <- as.numeric(actual==lv[2])
            
            f <- as.formula(paste("~", paste(feature_cols, collapse="+")))
            X <- model.matrix(f, df)[, -1, drop=FALSE]
            dtrain <- xgb.DMatrix(X, label=y_num)
            params <- list(objective="binary:logistic", eta=eta_val, max_depth=md_val)
            fit <- xgb.train(params=params, data=dtrain, nrounds=nr_val, verbose=0)
            prob <- predict(fit, newdata=X)
            pred_class <- ifelse(prob>=0.5, lv[2], lv[1])
            pred_class <- factor(pred_class, levels=lv)
            
            mets <- get_classification_metrics(actual, pred_class, prob)
            varimp <- get_var_importance_str(fit, mod_iname, df, target_col, colnames(X))
            
            current_models[[mod_iname]] <- list(booster=fit, x_names=colnames(X), y_levels=lv)
            predictions_list[[mod_iname]] <- pred_class
            all_metrics[[mod_iname]] <- list(
              ModelName=MODEL_NAME_MAP[mod_iname],
              MAE=NA, MSE=NA, RMSE=NA, R2=NA,
              Accuracy=round(mets$Accuracy,3),
              F1=round(mets$F1,3),
              AUC=round(mets$AUC,3),
              Hyperparams=paste0("eta=",eta_val,", max_depth=",md_val,", nrounds=",nr_val),
              VarImp=varimp
            )
            
          } else if(mod_iname=="classif.svm"){
            kernel_val <- ifelse(is.null(input$classif_svm_kernel), "radial", input$classif_svm_kernel)
            cost_val <- ifelse(is.null(input$classif_svm_cost), 1, input$classif_svm_cost)
            gamma_val <- ifelse(is.null(input$classif_svm_gamma), 0.1, input$classif_svm_gamma)
            
            f <- as.formula(paste(target_col, "~", paste(feature_cols, collapse="+")))
            fit <- e1071::svm(f, data=df, kernel=kernel_val, cost=cost_val,
                              gamma=gamma_val, type="C-classification")
            pred <- predict(fit, newdata=df)
            mets <- get_classification_metrics(actual, pred)
            varimp <- "Not Supported (svm)"
            
            current_models[[mod_iname]] <- fit
            predictions_list[[mod_iname]] <- pred
            all_metrics[[mod_iname]] <- list(
              ModelName=MODEL_NAME_MAP[mod_iname],
              MAE=NA, MSE=NA, RMSE=NA, R2=NA,
              Accuracy=round(mets$Accuracy,3),
              F1=round(mets$F1,3),
              AUC=round(mets$AUC,3),
              Hyperparams=paste0("kernel=",kernel_val,", cost=",cost_val,", gamma=",gamma_val),
              VarImp=varimp
            )
            
          } else if(mod_iname=="classif.log_reg"){
            lv <- levels(actual)
            if(length(lv)==2){
              # 2-class => glm
              f <- as.formula(paste(target_col, "~", paste(feature_cols, collapse="+")))
              fit <- glm(f, data=df, family=binomial)
              prob <- predict(fit, newdata=df, type="response")
              pred_class <- ifelse(prob>=0.5, lv[2], lv[1])
              pred_class <- factor(pred_class, levels=lv)
              
              mets <- get_classification_metrics(actual, pred_class, prob)
              varimp <- "Coefficients: summary(glm)"
              
              current_models[[mod_iname]] <- fit
              predictions_list[[mod_iname]] <- pred_class
              all_metrics[[mod_iname]] <- list(
                ModelName=MODEL_NAME_MAP[mod_iname],
                MAE=NA, MSE=NA, RMSE=NA, R2=NA,
                Accuracy=round(mets$Accuracy,3),
                F1=round(mets$F1,3),
                AUC=NA,
                Hyperparams="(2-class glm)",
                VarImp=varimp
              )
            } else {
              # Multi-class => multinom
              f <- as.formula(paste(target_col, "~", paste(feature_cols, collapse="+")))
              fit <- nnet::multinom(f, data=df, trace=FALSE)
              pred_class <- predict(fit, newdata=df)
              
              acc <- mean(actual==pred_class)
              varimp <- "Coefficients: summary(multinom)"
              
              current_models[[mod_iname]] <- fit
              predictions_list[[mod_iname]] <- factor(pred_class, levels=lv)
              all_metrics[[mod_iname]] <- list(
                ModelName=MODEL_NAME_MAP[mod_iname],
                MAE=NA, MSE=NA, RMSE=NA, R2=NA,
                Accuracy=round(acc,3),
                F1=NA, AUC=NA,
                Hyperparams="(multi-class)",
                VarImp=varimp
              )
            }
            
          } else if(mod_iname=="classif.naive_bayes"){
            f <- as.formula(paste(target_col, "~", paste(feature_cols, collapse="+")))
            fit <- e1071::naiveBayes(f, data=df)
            pred <- predict(fit, newdata=df)
            mets <- get_classification_metrics(actual, pred)
            varimp <- "Not Supported (naiveBayes)"
            
            current_models[[mod_iname]] <- fit
            predictions_list[[mod_iname]] <- pred
            all_metrics[[mod_iname]] <- list(
              ModelName=MODEL_NAME_MAP[mod_iname],
              MAE=NA, MSE=NA, RMSE=NA, R2=NA,
              Accuracy=round(mets$Accuracy,3),
              F1=round(mets$F1,3),
              AUC=NA,
              Hyperparams="(no params)",
              VarImp=varimp
            )
            
          } else if(mod_iname=="classif.kknn"){
            k_val <- ifelse(is.null(input$classif_kknn_k), 5, input$classif_kknn_k)
            f <- as.formula(paste(target_col, "~", paste(feature_cols, collapse="+")))
            fit <- kknn::train.kknn(f, data=df, kmax=k_val)
            pred <- predict(fit, newdata=df)
            mets <- get_classification_metrics(actual, pred)
            varimp <- "Not Supported (kknn)"
            
            current_models[[mod_iname]] <- fit
            predictions_list[[mod_iname]] <- pred
            all_metrics[[mod_iname]] <- list(
              ModelName=MODEL_NAME_MAP[mod_iname],
              MAE=NA, MSE=NA, RMSE=NA, R2=NA,
              Accuracy=round(mets$Accuracy,3),
              F1=round(mets$F1,3),
              AUC=NA,
              Hyperparams=paste0("kmax=",k_val),
              VarImp=varimp
            )
          }
        }
        
        # Classification Visualization
        df_pred <- data.frame(actual=actual)
        for(nm in names(predictions_list)){
          df_pred[[nm]] <- predictions_list[[nm]]
        }
        df_long <- pivot_longer(df_pred, cols=-actual, names_to="model", values_to="predicted")
        p <- ggplot(df_long, aes(x=actual, fill=model)) +
          geom_bar(position="dodge") +
          theme_minimal() +
          labs(title="Classification: Prediction Results", x="Class", y="Count")
        output$prediction_plot <- renderPlot({ p })
      }
      
      df_metrics <- do.call(rbind, lapply(names(all_metrics), function(mn){
        am <- all_metrics[[mn]]
        data.frame(
          ModelName=am$ModelName,
          MAE=am$MAE,
          MSE=am$MSE,
          RMSE=am$RMSE,
          R2=am$R2,
          Accuracy=am$Accuracy,
          F1=am$F1,
          AUC=am$AUC,
          Hyperparams=am$Hyperparams,
          VarImp=am$VarImp,
          stringsAsFactors=FALSE
        )
      }))
      model_metrics_data(df_metrics)
      
      if(task_type()=="Regression"){
        df_metrics1 <- df_metrics[, c("ModelName","MAE","MSE","RMSE","R2")]
      } else {
        df_metrics1 <- df_metrics[, c("ModelName","Accuracy","F1","AUC")]
      }
      df_metrics2 <- df_metrics[, c("ModelName","Hyperparams","VarImp")]
      
      output$model_metrics_table1 <- DT::renderDataTable({
        df_metrics1
      }, options=list(scrollX=TRUE, pageLength=10))
      
      output$model_metrics_table2 <- DT::renderDataTable({
        df_metrics2
      }, options=list(scrollX=TRUE, pageLength=10))
      
      trained_models(current_models)
      model_choices <- as.character(MODEL_NAME_MAP[names(current_models)])
      updateSelectInput(session, "model_for_prediction",
                        choices=c("All", model_choices),
                        selected="All")
      updateSelectInput(session, "inverse_model",
                        choices=model_choices)
    })
  })
  
  # New Data Prediction
  prediction_results <- reactiveVal(NULL)
  new_data <- reactiveVal(data.frame())
  
  observeEvent(list(input$target_var, input$features_var, input$newdata_input_method), {
    # Default for manual input
    if(input$newdata_input_method=="Upload CSV") return()
    if(!is.null(input$features_var) && length(input$features_var)>0){
      empty_df <- data.frame(matrix(nrow=1, ncol=length(input$features_var)))
      colnames(empty_df) <- input$features_var
      new_data(empty_df)
      output$new_data_table <- renderRHandsontable({
        rhandsontable(empty_df, rowHeaders=NULL, useTypes=FALSE, width="100%", height=300)
      })
    }
  })
  
  new_prediction_data <- reactive({
    if(input$newdata_input_method=="Upload CSV") {
      if(!is.null(input$newdata_file)){
        df_new <- tryCatch({
          read.csv(input$newdata_file$datapath, header=TRUE, stringsAsFactors=FALSE)
        }, error=function(e){
          showNotification(paste("New prediction CSV reading error:", e$message), type="error")
          return(NULL)
        })
        return(df_new)
      } else {
        return(NULL)
      }
    } else {
      if(!is.null(input$new_data_table)){
        df_manual <- hot_to_r(input$new_data_table)
        return(df_manual)
      }
      return(new_data())
    }
  })
  
  convert_newdata_types <- function(df_new, df_train){
    if(is.null(df_new) || nrow(df_new)==0) return(df_new)
    for(col in colnames(df_new)){
      if(col %in% colnames(df_train)){
        if(is.factor(df_train[[col]])){
          lv_train <- levels(df_train[[col]])
          bad_idx <- which(! df_new[[col]] %in% lv_train)
          if(length(bad_idx)>0){
            showNotification(paste("Column",col,"has invalid categories:",
                                   paste(unique(df_new[[col]][bad_idx]), collapse=",")), type="warning")
            df_new[[col]][bad_idx] <- NA
          }
          df_new[[col]] <- factor(df_new[[col]], levels=lv_train)
        } else if(is.numeric(df_train[[col]])){
          conv_val <- suppressWarnings(as.numeric(as.character(df_new[[col]])))
          bad_idx <- which(is.na(conv_val) & !is.na(df_new[[col]]))
          if(length(bad_idx)>0){
            showNotification(paste("Numeric conversion failed in column",col,":",
                                   paste(unique(df_new[[col]][bad_idx]),collapse=",")), type="warning")
          }
          df_new[[col]] <- conv_val
        }
      }
    }
    df_new2 <- df_new[complete.cases(df_new), , drop=FALSE]
    if(nrow(df_new2)<nrow(df_new)){
      showNotification(paste(nrow(df_new)-nrow(df_new2),"row(s) removed due to invalid values."), type="warning")
    }
    df_new2
  }
  
  safe_predict_model <- function(iname, model_obj, df_new, df_train, target_var){
    out <- NULL
    tryCatch({
      if(grepl("regr.rpart", iname)){
        out <- predict(model_obj, newdata=df_new)
      } else if(grepl("regr.lm", iname)){
        out <- predict(model_obj, newdata=df_new)
      } else if(grepl("regr.rf", iname)){
        out <- predict(model_obj, newdata=df_new)
      } else if(grepl("regr.xgboost", iname)){
        f <- as.formula(paste("~", paste(input$features_var, collapse="+")))
        X_new <- model.matrix(f, df_new)[, -1, drop=FALSE]
        out <- predict(model_obj, newdata=X_new)
      } else if(grepl("regr.lasso_cv", iname)){
        f <- as.formula(paste("~", paste(input$features_var, collapse="+")))
        X_new <- model.matrix(f, df_new)[, -1, drop=FALSE]
        pval <- predict(model_obj, newx=X_new, s="lambda.min")
        out <- as.numeric(pval)
      } else if(grepl("classif.rpart", iname)){
        out <- predict(model_obj, newdata=df_new, type="class")
      } else if(grepl("classif.rf", iname)){
        out <- predict(model_obj, newdata=df_new)
      } else if(grepl("classif.xgboost", iname)){
        X_new <- model.matrix(as.formula(paste("~", paste(input$features_var, collapse="+"))), df_new)[, -1, drop=FALSE]
        prob <- predict(model_obj$booster, newdata=X_new)
        lv <- model_obj$y_levels
        out_class <- ifelse(prob>=0.5, lv[2], lv[1])
        out <- factor(out_class, levels=lv)
      } else if(grepl("classif.svm", iname)){
        out <- predict(model_obj, newdata=df_new)
      } else if(grepl("classif.log_reg", iname)){
        lv <- levels(df_train[[target_var]])
        if(length(lv)==2){
          prob <- predict(model_obj, newdata=df_new, type="response")
          out_class <- ifelse(prob>=0.5, lv[2], lv[1])
          out <- factor(out_class, levels=lv)
        } else {
          out_class <- predict(model_obj, newdata=df_new)
          out <- factor(out_class, levels=lv)
        }
      } else if(grepl("classif.naive_bayes", iname)){
        out <- predict(model_obj, newdata=df_new)
      } else if(grepl("classif.kknn", iname)){
        out <- predict(model_obj, newdata=df_new)
      }
    }, error=function(e){
      showNotification(paste("Prediction failed (model:", iname, "):", e$message), type="error")
      out <<- NULL
    })
    out
  }
  
  observeEvent(input$predict_new, {
    if(length(trained_models())==0){
      showNotification("Please train a model first.", type="error")
      return()
    }
    withProgress(message="Running prediction...", value=0, {
      allmods <- trained_models()
      if(is.null(input$model_for_prediction)){
        showNotification("No model selected for prediction.", type="error")
        return()
      }
      choice <- input$model_for_prediction
      
      df_train <- get_filtered_data()
      if(is.null(df_train) || nrow(df_train)==0){
        showNotification("No training data available.", type="error")
        return()
      }
      
      df_new <- new_prediction_data()
      if(is.null(df_new) || nrow(df_new)==0){
        showNotification("New data is empty.", type="error")
        return()
      }
      df_new2 <- convert_newdata_types(df_new, df_train)
      if(nrow(df_new2)==0){
        showNotification("No valid new data rows (type conversion failure etc.).", type="error")
        return()
      }
      
      incProgress(0.3, detail="Formatting input data...")
      pred_df <- df_new2
      
      if(choice=="All"){
        model_keys <- names(allmods)
        for(mk in model_keys){
          incProgress(0.7/length(model_keys), detail=paste("Predicting using:", MODEL_NAME_MAP[mk]))
          pvec <- safe_predict_model(mk, allmods[[mk]], df_new2, df_train, input$target_var)
          if(!is.null(pvec)){
            gen_name <- MODEL_NAME_MAP[mk]
            pred_df[[paste0("Prediction_", gen_name)]] <- pvec
          }
        }
      } else {
        mk <- names(MODEL_NAME_MAP)[MODEL_NAME_MAP==choice]
        if(length(mk)==0){
          showNotification("The selected model was not found.", type="error")
          return()
        }
        pvec <- safe_predict_model(mk, allmods[[mk]], df_new2, df_train, input$target_var)
        if(!is.null(pvec)){
          pred_df[[paste0("Prediction_", choice)]] <- pvec
        }
      }
      
      output$new_data_table <- renderRHandsontable({
        rhandsontable(pred_df, rowHeaders=NULL, useTypes=FALSE)
      })
      prediction_results(pred_df)
    })
  })
  
  output$download_pred <- downloadHandler(
    filename=function(){
      paste0("predicted_results_", Sys.Date(), ".csv")
    },
    content=function(file){
      res <- prediction_results()
      if(!is.null(res)){
        write.csv(res, file, row.names=FALSE)
      }
    }
  )
  
  # Inverse Prediction (GA)
  inverse_results <- reactiveVal(NULL)
  
  output$target_value_ui <- renderUI({
    df <- get_filtered_data()
    if(is.null(df)) return(NULL)
    if(task_type()=="Regression"){
      numericInput("target_value", "Target Variable Goal Value", value=0)
    } else {
      lv <- levels(df[[input$target_var]])
      selectInput("target_value", "Target Variable Goal Value", choices=lv)
    }
  })
  
  output$var_range_ui <- renderUI({
    df <- get_filtered_data()
    if(is.null(df) || is.null(input$features_var)) return(NULL)
    
    lapply(input$features_var, function(f){
      ft <- input[[paste0("feature_type_", f)]]
      if(!is.null(ft) && ft=="Numeric"){
        tagList(
          h5(f),
          fluidRow(
            column(4, numericInput(paste0("min_", f), "Minimum", value=min(df[[f]], na.rm=TRUE))),
            column(4, numericInput(paste0("max_", f), "Maximum", value=max(df[[f]], na.rm=TRUE))),
            column(4, numericInput(paste0("interval_", f), "Interval",
                                   value=round((max(df[[f]], na.rm=TRUE)-min(df[[f]], na.rm=TRUE))/10,3),
                                   min=0))
          )
        )
      } else {
        lv <- if(!is.null(df[[f]])) levels(df[[f]]) else c()
        tagList(
          h5(f),
          selectInput(paste0("levels_", f), "Select Categories", choices=lv, multiple=TRUE)
        )
      }
    })
  })
  
  observeEvent(input$run_ga, {
    if(is.null(input$inverse_model) || is.null(input$features_var)){
      showNotification("Missing items required for inverse prediction.", type="error")
      return()
    }
    
    for (f in input$features_var) {
      ft <- input[[paste0("feature_type_", f)]]
      if (ft == "Numeric") {
        if (is.null(input[[paste0("min_", f)]]) || is.null(input[[paste0("max_", f)]]) ||
            is.null(input[[paste0("interval_", f)]])) {
          showNotification(paste("For numeric variable", f, "Minimum, Maximum, and Interval are not entered."), type = "error")
          return()
        }
      } else {
        if (is.null(input[[paste0("levels_", f)]])) {
          showNotification(paste("For factor variable", f, "Please enter the categories to use."), type = "error")
          return()
        }
      }
    }
    
    model_name <- names(MODEL_NAME_MAP)[MODEL_NAME_MAP==input$inverse_model]
    if(is.null(model_name) || length(model_name)==0){
      showNotification("The selected model was not found.", type="error")
      return()
    }
    mods <- trained_models()
    if(!model_name %in% names(mods)){
      showNotification("The selected model has not been trained.", type="error")
      return()
    }
    model_obj <- mods[[model_name]]
    
    df_train <- get_filtered_data()
    if(is.null(df_train) || nrow(df_train)==0){
      showNotification("No training data available.", type="error")
      return()
    }
    
    withProgress(message="Running Inverse Prediction...", value=0, {
      incProgress(0.2, detail="Retrieving model and parameters...")
      
      gene_lower <- c()
      gene_upper <- c()
      for(f in input$features_var){
        ft <- input[[paste0("feature_type_", f)]]
        if(!is.null(ft) && ft=="Numeric"){
          gene_lower <- c(gene_lower, input[[paste0("min_", f)]])
          gene_upper <- c(gene_upper, input[[paste0("max_", f)]])
        } else {
          lv <- input[[paste0("levels_", f)]]
          gene_lower <- c(gene_lower, 1)
          gene_upper <- c(gene_upper, length(lv))
        }
      }
      
      fitness_func <- function(x){
        new_data0 <- list()
        idx <- 1
        for(f in input$features_var){
          ft <- input[[paste0("feature_type_", f)]]
          if(!is.null(ft) && ft=="Numeric"){
            min_val <- input[[paste0("min_", f)]]
            interval_val <- input[[paste0("interval_", f)]]
            val <- x[idx]
            new_val <- min_val + round((val-min_val)/interval_val)*interval_val
            new_val <- max(new_val, input[[paste0("min_", f)]])
            new_val <- min(new_val, input[[paste0("max_", f)]])
            new_data0[[f]] <- new_val
          } else {
            lv <- input[[paste0("levels_", f)]]
            idx_val <- round(x[idx])
            idx_val <- max(min(idx_val, length(lv)),1)
            new_data0[[f]] <- factor(lv[idx_val], levels=lv)
          }
          idx <- idx+1
        }
        new_data0 <- as.data.frame(new_data0, stringsAsFactors=FALSE)
        new_data2 <- convert_newdata_types(new_data0, df_train)
        if(nrow(new_data2)==0) return(-Inf)
        
        pred_val <- safe_predict_model(model_name, model_obj, new_data2, df_train, input$target_var)
        if(is.null(pred_val)) return(-Inf)
        
        if(task_type()=="Regression"){
          goal <- as.numeric(input$target_value)
          return(-abs(pred_val - goal))
        } else {
          goal_class <- input$target_value
          if(length(pred_val)==1 && pred_val==goal_class) return(1) else return(0)
        }
      }
      
      ga_result <- GA::ga(
        type="real-valued",
        fitness=fitness_func,
        lower=gene_lower,
        upper=gene_upper,
        popSize=50, 
        maxiter=100,
        pmutation=0.1,
        pcrossover=0.8,
        keepBest=TRUE
      )
      
      incProgress(0.2, detail="Processing results...")
      pop <- ga_result@population
      pop_fitness <- apply(pop,1, fitness_func)
      pop_df_raw <- as.data.frame(pop)
      colnames(pop_df_raw) <- input$features_var
      pop_df_raw$fitness <- pop_fitness
      pop_df_sorted <- pop_df_raw[order(-pop_df_raw$fitness), ]
      
      processed_candidates <- lapply(1:nrow(pop_df_sorted), function(i){
        rowv <- pop_df_sorted[i, input$features_var, drop=FALSE]
        cand_list <- list()
        idx <- 1
        for(f in input$features_var){
          ft <- input[[paste0("feature_type_", f)]]
          val <- rowv[[f]]
          if(ft=="Numeric"){
            min_val <- input[[paste0("min_", f)]]
            interval_val <- input[[paste0("interval_", f)]]
            pval <- min_val + round((val-min_val)/interval_val)*interval_val
            pval <- min(max(pval, input[[paste0("min_", f)]]), input[[paste0("max_", f)]])
            cand_list[[f]] <- pval
          } else {
            lv <- input[[paste0("levels_", f)]]
            idx_val <- round(val)
            idx_val <- max(min(idx_val, length(lv)),1)
            cand_list[[f]] <- factor(lv[idx_val], levels=lv)
          }
          idx <- idx+1
        }
        dfcand <- as.data.frame(cand_list, stringsAsFactors=FALSE)
        dfcand$fitness <- pop_df_sorted$fitness[i]
        dfcand
      })
      processed_df <- do.call(rbind, processed_candidates)
      processed_df_unique <- processed_df[!duplicated(processed_df[, input$features_var]), ]
      processed_df_unique <- processed_df_unique[order(-processed_df_unique$fitness), ]
      n_sol <- min(input$n_solutions, nrow(processed_df_unique))
      final_candidates <- processed_df_unique[1:n_sol, ]
      final_candidates$fitness <- NULL
      
      # Recalculate prediction values
      final_candidates$PredictedValue <- sapply(1:nrow(final_candidates), function(i){
        candrow <- final_candidates[i, input$features_var, drop=FALSE]
        candrow2 <- convert_newdata_types(candrow, df_train)
        if(nrow(candrow2)==0) return(NA)
        out <- safe_predict_model(model_name, model_obj, candrow2, df_train, input$target_var)
        if(is.null(out)) return(NA)
        out
      })
      
      output$ga_results_table <- DT::renderDataTable({
        DT::datatable(final_candidates, options=list(scrollX=TRUE, pageLength=5), caption="Optimal Solution Candidates")
      })
      inverse_results(final_candidates)
    })
  })
  
  # Similarity Calculation
  similarity_report_data <- reactiveVal(NULL)
  
  output$similarity_selection_ui <- renderUI({
    req(data_reactive(), input$name_var)
    if(input$similarity_method == "Select by Name"){
      df <- data_reactive()
      choices <- unique(df[[input$name_var]])
      selectInput("similarity_name", "Select Name", choices = choices)
    } else {
      # Manual Input: Display rHandsontable
      if(is.null(input$features_var)) return(NULL)
      tagList(
        rHandsontableOutput("similarity_manual_table")
      )
    }
  })
  
  output$similarity_manual_table <- renderRHandsontable({
    req(input$features_var)
    df <- data_reactive()
    # For each feature: if numeric then use mean, if factor then use first level
    default_values <- sapply(input$features_var, function(var) {
      if(is.numeric(df[[var]])){
        mean(df[[var]], na.rm = TRUE)
      } else {
        levels(as.factor(df[[var]]))[1]
      }
    })
    manual_df <- as.data.frame(as.list(default_values), stringsAsFactors = FALSE)
    # Ensure type consistency
    for(var in input$features_var){
      if(is.numeric(df[[var]])){
        manual_df[[var]] <- as.numeric(manual_df[[var]])
      } else {
        manual_df[[var]] <- as.character(manual_df[[var]])
      }
    }
    rhandsontable(manual_df, rowHeaders = FALSE)
  })
  
  observeEvent(input$find_similarity, {
    req(data_reactive(), input$name_var, input$features_var)
    df <- data_reactive()
    
    get_normalized_value <- function(value, var, df) {
      if(is.numeric(df[[var]])){
        rng <- range(df[[var]], na.rm = TRUE)
        if(diff(rng)==0) return(0.5)
        return((value - rng[1]) / diff(rng))
      } else {
        lev <- levels(as.factor(df[[var]]))
        num <- as.numeric(factor(value, levels = lev))
        if(length(lev)==1) return(0.5)
        return((num - 1) / (length(lev) - 1))
      }
    }
    
    query_vector <- sapply(input$features_var, function(var){
      if(input$similarity_method == "Select by Name"){
        selected_row <- df[df[[input$name_var]] == input$similarity_name, ]
        if(nrow(selected_row)==0){
          showNotification("No data found for the selected name.", type="error")
          return(NA)
        }
        get_normalized_value(selected_row[[var]][1], var, df)
      } else {
        # For manual input: get values from rhandsontable
        manual_input <- hot_to_r(input$similarity_manual_table)
        get_normalized_value(manual_input[[var]][1], var, df)
      }
    })
    
    feature_matrix <- t(sapply(1:nrow(df), function(i){
      sapply(input$features_var, function(var){
        get_normalized_value(df[i, var], var, df)
      })
    }))
    
    cosine_similarity <- function(x, y){
      sum(x * y) / (sqrt(sum(x * x)) * sqrt(sum(y * y)) + 1e-10)
    }
    similarities <- apply(feature_matrix, 1, function(row) cosine_similarity(row, query_vector))
    df$CosineSimilarity <- round(similarities,3)
    
    top_n <- head(df[order(-df$CosineSimilarity), ], n = input$candidate_num)
    
    output$similarity_table <- DT::renderDataTable({
      DT::datatable(top_n, options = list(scrollX = TRUE, pageLength = input$candidate_num))
    })
    
    # Query data for plot
    query_df <- data.frame(
      Feature = input$features_var,
      Value = as.numeric(query_vector[input$features_var]),
      Source = "Query",
      LineType = "solid",
      stringsAsFactors = FALSE
    )
    
    # Candidate data: calculate normalized values for each candidate row
    candidate_df <- do.call(rbind, lapply(1:nrow(top_n), function(i) {
      candidate_row <- top_n[i, ]
      data.frame(
        Feature = input$features_var,
        Value = sapply(input$features_var, function(var) {
          get_normalized_value(candidate_row[[var]], var, df)
        }),
        Source = paste0("Candidate: ", 
                        if(!is.null(input$name_var)) as.character(candidate_row[[input$name_var]]) else i),
        LineType = "dotted",
        stringsAsFactors = FALSE
      )
    }))
    
    plot_df <- rbind(query_df, candidate_df)
    plot_df$Feature <- factor(plot_df$Feature, levels = input$features_var)
    
    output$similarity_plot <- renderPlot({
      ggplot(plot_df, aes(x = Feature, y = Value, group = Source, color = Source, linetype = LineType)) +
        geom_line () +
        geom_point() +
        labs(title = "Comparison of Feature Variables", y = "Normalized Value", x = "Feature Variables") +
        theme_minimal() +
        scale_linetype_manual(values = c("solid" = "solid", "dotted" = "dotted"))
    })
    
    # --- Prepare Report Data Table ---
    if(input$similarity_method == "Select by Name"){
      query_orig <- df[df[[input$name_var]] == input$similarity_name, ][1, ]
      report_query <- data.frame(
        Source = "QUERY",
        Cos = NA,
        query_orig[, input$features_var, drop = FALSE],
        stringsAsFactors = FALSE
      )
    } else {
      manual_input <- hot_to_r(input$similarity_manual_table)
      report_query <- data.frame(
        Source = "QUERY",
        Cos = NA,
        manual_input[, input$features_var, drop = FALSE],
        stringsAsFactors = FALSE
      )
    }
    
    report_candidate <- top_n[, c(if(!is.null(input$name_var)) input$name_var, input$features_var, "CosineSimilarity"), drop = FALSE]
    if(!is.null(input$name_var) && input$name_var %in% names(report_candidate)){
      names(report_candidate)[names(report_candidate)==input$name_var] <- "Source"
    } else {
      report_candidate$Source <- paste0("Candidate", seq_len(nrow(report_candidate)))
    }
    names(report_candidate)[names(report_candidate)=="CosineSimilarity"] <- "Cos"
    
    similarity_report_table <- rbind(report_query, report_candidate)
    
    similarity_report_data(similarity_report_table)
    
  })
  
  
  # Report Output
  reportContent <- eventReactive(input$generate_report, {
    df <- get_filtered_data()
    if(is.null(df)) {
      return("No data available.")
    }
    file_name <- if(!is.null(input$file$name)) input$file$name else "Not Specified"
    data_summary_text <- capture.output({
      if(nrow(df)>0) skim(df) else cat("No data available.")
    })
    
    tv <- input$target_var
    fv <- input$features_var
    if(is.null(fv)) fv <- character(0)
    
    # Target vs Feature
    target_feature_text <- ""
    if(length(fv)>0){
      assoc_vec <- sapply(fv, function(x){
        round(measure_association(df[[tv]], df[[x]]), 3)
      })
      target_feature_text <- paste(capture.output(print(data.frame(Association=assoc_vec))), collapse="\n")
    } else {
      target_feature_text <- "No feature variables selected."
    }
    
    # Feature Associations
    if(length(fv)>=2){
      assoc_mat <- matrix(NA, nrow=length(fv), ncol=length(fv), dimnames=list(fv,fv))
      for(i in seq_along(fv)){
        for(j in seq_along(fv)){
          assoc_mat[i,j] <- measure_association(df[[fv[i]]], df[[fv[j]]])
        }
      }
      assoc_mat <- round(assoc_mat,3)
      features_assoc_text <- paste(capture.output(print(as.data.frame(assoc_mat))), collapse="\n")
    } else {
      features_assoc_text <- "At least 2 feature variables are required."
    }
    
    # Similarity Section
    similarity_section <- ""
    if(!is.null(similarity_report_data())){
      sim_table <- similarity_report_data()
      sim_table_text <- paste(capture.output(print(sim_table)), collapse="\n")
      similarity_section <- paste0(
        "【Similarity Calculation Results】\n",
        sim_table_text, "\n\n"
      )
    }
    
    
    mm_data <- model_metrics_data()
    metric_text_1 <- ""
    metric_text_2 <- ""
    if(is.null(mm_data)){
      metric_text_1 <- "No models have been trained yet."
    } else {
      if(task_type()=="Regression"){
        sub_metrics <- mm_data[, c("ModelName","MAE","MSE","RMSE","R2")]
      } else {
        sub_metrics <- mm_data[, c("ModelName","Accuracy","F1","AUC")]
      }
      sub_detail <- mm_data[, c("ModelName","Hyperparams","VarImp")]
      metric_text_1 <- paste(capture.output(print(sub_metrics)), collapse="\n")
      metric_text_2 <- paste(capture.output(print(sub_detail)), collapse="\n")
    }
    
    prediction_text <- ""
    if(!is.null(prediction_results()) && nrow(prediction_results())>0){
      prediction_text <- "【Prediction】 New data prediction results are available for download.\n\n"
    }
    
    target_range_or_cats <- ""
    if(!is.null(input$target_var_type) && input$target_var_type=="Numeric"){
      rng <- range(df[[tv]], na.rm=TRUE)
      target_range_or_cats <- paste0("[", rng[1]," to ", rng[2],"]")
    } else {
      if(!is.null(df[[tv]])){
        lv <- levels(df[[tv]])
        target_range_or_cats <- paste0("Categories: ", paste(lv, collapse=", "))
      }
    }
    
    features_range_or_cats <- ""
    if(length(fv)>0){
      features_range_or_cats <- paste(
        sapply(fv, function(f){
          ft <- input[[paste0("feature_type_", f)]]
          if(ft=="Numeric"){
            rng <- range(df[[f]], na.rm=TRUE)
            paste0(f, ": [", rng[1], " to ", rng[2], "]")
          } else {
            lv <- levels(df[[f]])
            paste0(f, ": Categories=", paste(lv, collapse=", "))
          }
        }),
        collapse="\n"
      )
    } else {
      features_range_or_cats <- "No feature variables selected."
    }
    
    inverse_text <- ""
    if(!is.null(inverse_results())){
      inverse_text <- paste0(
        "【Inverse Prediction】\n",
        "Model Used: ", input$inverse_model, "\n",
        "Goal Value: ", input$target_value, "\n",
        "Inverse Prediction Results:\n",
        paste(capture.output(print(inverse_results())), collapse="\n"),
        "\n\n"
      )
    }
    
    paste0(
      "**", app_name, "**\n\n",
      "【Report Generated At】", Sys.time(), "\n\n",
      "【Dataset】", file_name, "\n\n",
      "【Task Type】", task_type(), "\n\n",
      "【Target Variable】", input$target_var, " (", input$target_var_type, ")", "\n",
      "【Target Variable Range/Categories】", target_range_or_cats, "\n\n",
      "【Feature Variables】", if(length(fv)>0){ 
        paste(sapply(fv, function(x) paste0(x," (", input[[paste0("feature_type_", x)]],")")), collapse=", ")
      } else {"None"}, "\n",
      "【Feature Variables Range/Categories】\n", features_range_or_cats, "\n\n",
      "【Data Summary】\n", paste(data_summary_text, collapse="\n"), "\n\n",
      "【Correlation between Target and Features】\n", target_feature_text, "\n\n",
      "【Association among Features】\n", features_assoc_text, "\n",
      "  (Notes):\n",
      "- Numeric vs Numeric → Pearson correlation (-1 to 1)\n",
      "- Factor vs Factor → Cramér's V (0 to 1)\n",
      "- Numeric vs Factor → Correlation Ratio (η) (0 to 1)\n\n",
      similarity_section,
      "【Trained Model (Hyperparams, VarImp)】\n", metric_text_2, "\n\n",
      "【Training Results (Metrics)】\n", metric_text_1, "\n\n",
      prediction_text,
      inverse_text
    )
  })
  
  output$report_output <- renderText({
    reportContent()
  })
  
  # Report Download
  output$download_report <- downloadHandler(
    filename=function(){
      paste0("MachineLearning_Report_", Sys.Date(), ".txt")
    },
    content=function(file){
      cat(reportContent(), file=file)
    }
  )
  
  # Model Explanation
  output$model_explanation <- renderUI({
    HTML("
      <h3>Explanation of Learning Models</h3>
      <h4>[Regression Models]</h4>
      <li>
        <strong>Decision Tree (regr)</strong><br>
        Splits data based on conditions to predict the target variable.<br>
        <em>cp</em> (complexity parameter) is a threshold: if the reduction in error from a split is less than this value, the split is not made.<br>
        A larger cp results in fewer splits and a simpler tree, while a smaller cp results in more detailed splits, increasing training accuracy but with a higher risk of overfitting.<br>
        Variable importance is computed as the total error reduction from each split.
      </li>
      <li>
        <strong>Linear Regression (regr)</strong><br>
        A basic linear model given by the equation:<br>
        y = β<sub>0</sub> + β<sub>1</sub>x<sub>1</sub> + … + β<sub>p</sub>x<sub>p</sub><br>
        Coefficients β are estimated using the least squares method. There are no hyperparameters, and the influence of each variable is determined by the size and statistical significance of the coefficients.
      </li>
      <li>
        <strong>Random Forests (regr)</strong><br>
        Constructs multiple decision trees and averages their predictions.<br>
        <em>mtry</em> is the number of variables tried at each split; too high a value can lead to similar trees and overfitting.<br>
        <em>num.trees</em> is the total number of trees; more trees yield more stable predictions but increase computational cost.<br>
        Variable importance is evaluated by averaging the impurity reduction (or split contribution) across trees.
      </li>
      <li>
        <strong>xgboost (regr)</strong><br>
        Sequentially adds weak learners (typically decision trees) to correct the residuals from previous trees.<br>
        <em>eta</em> (learning rate) adjusts each tree’s influence on the final prediction.<br>
        <em>max_depth</em> is the maximum depth of each tree.<br>
        <em>nrounds</em> is the number of boosting rounds.<br>
        Variable importance is determined using metrics such as information gain.
      </li>
      <li>
        <strong>LASSO Regression (regr.lasso_cv)</strong><br>
        A linear regression model with an L1 regularization term.<br>
        <em>nfolds</em> indicates the number of folds for cross-validation.<br>
        Variable importance is evaluated based on the magnitude of the non-zero coefficients.
      </li>
  
      <h4>[Classification Models]</h4>
      <li>
        <strong>Decision Tree (class)</strong><br>
        Splits data based on class label impurity to perform classification.<br>
        <em>cp</em> is the threshold for impurity reduction needed to make a split.<br>
        Variable importance is calculated as the total reduction in impurity.
      </li>
      <li>
        <strong>Random Forests (class)</strong><br>
        Determines the class by majority vote among multiple decision trees.<br>
        The settings for <em>mtry</em> and <em>num.trees</em> are crucial.<br>
        Variable importance is evaluated based on impurity reduction and contribution to overall accuracy.
      </li>
      <li>
        <strong>xgboost (class)</strong><br>
        This implementation supports only 2-class classification.<br>
        Requires setting <em>eta</em>, <em>max_depth</em>, and <em>nrounds</em> for classification.<br>
        Variable importance is determined using metrics such as information gain.
      </li>
      <li>
        <strong>Support Vector Machine (svm, class)</strong><br>
        Does not support factor-type feature variables.<br>
        The primary parameters are <em>kernel</em>, <em>cost</em>, and <em>gamma</em>.<br>
        Variable importance is not directly computed.
      </li>
      <li>
        <strong>Logistic Regression (class)</strong><br>
        Uses a standard GLM for 2-class problems and multinom from the nnet package for multi-class problems.<br>
        No hyperparameters are set; the influence of variables is evaluated from the coefficients.
      </li>
      <li>
        <strong>Naive Bayes (class)</strong><br>
        Computes class probabilities using conditional probabilities.<br>
        Variable importance is not directly calculated.
      </li>
      <li>
        <strong>k-Nearest Neighbors (class)</strong><br>
        Determines the class by majority vote among the nearest neighbors.<br>
        A clear measure of variable importance is not provided.
      </li>
    ")
  })
}

shinyApp(ui, server)

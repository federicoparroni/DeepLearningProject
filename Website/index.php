<html> 
<head>   

   <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">

   <link href="css/dropzone base.css" type="text/css" rel="stylesheet" />
   <link href="css/dropzone custom.css" type="text/css" rel="stylesheet" />

   <link href="css/app.css" type="text/css" rel="stylesheet" />

   <script src="js/dropzone.js"></script>
   
   <title>GDP corp</title>
</head>
 
<body>

   <?php include 'header.php' ?>

   <section>

      <div class="card">
         <div class="card-body">
            <h5 class="card-title">Card title</h5>
            <p class="card-text">Salve a tutti!</br >
               <p class="my-3">Siamo tre studenti del Politecnico di Milano, stiamo raccogliendo immagini allo scopo di sviluppare un sistema per il face matching basato sul machine learning che potrà essere usato in diverse applicazioni, come lo sblocco di vari dispositivi attraverso il tuo viso.
               Per raggiungere questo obbiettivo abbiamo bisogno del tuo aiuto!</p>

               <p class="my-3">Abbiamo bisogno qualche immagine del tuo volto (10 immagini) in pose leggermente differenti (come mostrato nella figura sottostante).
               Puoi indossare occhiali, sorridere farie smorfie e scattare foto con diversi sfondi e condizioni di illuminazione, ma ricordati che i tratti fondamentali del tuo volto (occhi, bocca, naso ...) dovranno essere chiaramente visibili.</p>

               <p class="my-3">Se vuoi, puoi ripetere la compilazione di questa form più volte in modo da aiutarci di più!</p>

               <p class="my-3">Il nome lasciato durante la compilazione verrà citato nel paper finale del progetto.</p>
            </p>
            <!-- <p class="card-text"><small class="text-muted">Last updated 3 mins ago</small></p> -->
         </div>
         <div class="row justify-content-center">
            <img src="https://i.pinimg.com/originals/28/de/ed/28deedab62eee58b13cf815b521e8398.jpg" class="photo-example card-img-bottom col-md-8 col-xs-12" alt="Face poses example"></div>
         </div>
      </div>

      <form id="form" action="upload.php">
         <div class="form-group">
            <label for="name" class="py-2">Name (or nickname): </label>
            <input type="text" name="name" class="form-control" />

            <div id="dz" class="dropzone">
               <div class="dz-message">Tap here to upload your photos</div>
            </div>

            <button type="submit" class="btn btn-sm btn-outline-primary d-block mx-auto">Send your images</button>
         </div>
      </form>

      <div class="alert alert-danger d-none" role="alert">
         Some error occurred!
      </div>
      
   </section>

   <script src="js/app.js"></script>
</body>
</html>
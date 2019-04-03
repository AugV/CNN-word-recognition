# No Waste

## Description
Users will be able to store products into their product "basket" from 'TheMealDB' food database. Additionally users will be able to set their chosen products expiration date, to keep track of the foods which are about to go bad and suggesting recipes thus reducing waste of food. 

## Entity definition
* Product
    - Product ID - SHA-2 string
    - Product name -  string, length 50 characters
    - Description - string, length 200 characters
    - Creation date - ISO8601 string
    - Modification date - ISO8601 string
    - Expiry date - ISO8601 string
    - Category - string, length 50 characters
    - Priority - ISO 80000-2 number, 16-bit signed

## API definition
* GET method for the list of all ingredients in 'TheMealDB'. List stored in Backend for further selection by User.
    - GET https://www.themealdb.com/api/json/v1/1/list.php?i=list
* GET method for selection of specific product from ingredient list stored in database.
    - GET /api/1/product/:productId
    - {"ID":"1", "Product name": "Chicken", "Description": "Chicken is an ingredient extracted from chicken - the bird with feathers, with missing functionality of flying", "Category": "Meat"}
    - 404 Not Found
* GET method for displaying all ingredients in 'basket'.
    - GET /api/1/basket/list
    - {"Products": [{product1...}, {product2...},{product3...}]}
* PUT method for storing ingredient in 'basket'.
    - PUT /api/1/basket/:productId
    - {"ID":"1", "Product name": "Chicken", "Description": "Chicken is an ingredient extracted from chicken - the bird with feathers, with missing functionality of flying", "Creation date": "2019-03-29", "Modification date": "2019-03-29", "Expiry date": "2019-04-05", "Category": "Meat", "Priority": "1"}
* DELETE method for deleting product from 'basket'.
    - DELETE /api/1/basket/:productId
    - 404 Not Found
* POST method for updating product in 'basket'.
    - POST /api/1/basket/:productId
    - 404 Not Found

## UI definition
* Product list View: 
    - Displays a list of product currently in users basket.
    - Button which redirects to 'Product add' View.
    - When pressed on list object - two options, 'remove' and 'extend' appear.
    - https://wireframe.cc/usPET8 
* Product add View:
    - Displays a text entry field with dynamic list, which listens to users input and updates itself with values matching that input.
    - When pressed on list object - input for 'days until expiry' and 'Add' button appear.
    - https://wireframe.cc/6bHQcU
